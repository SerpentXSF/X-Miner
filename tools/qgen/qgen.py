# Quantus Poseidon2 (Goldilocks, width 12) as generated sm_86 SASS: field-op library, permutation emitter and the
# build of kernels/prebuilt/quantus_sm_86.cubin (see README.md; usage: qgen.py --cuassembler <dir> [--ccbin <dir>]).
# Every op is a list of instruction strings over fixed registers; sass.py orders them and adds stall counts.
# Lazy arithmetic like the CUDA kernel: values are any u64 congruent mod p; rare (~2^-32) slips are accepted.
import argparse, hashlib, os, re, random, shutil, subprocess, sys, tempfile
import sass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
P = 0xFFFFFFFF00000001
M32 = 0xFFFFFFFF

# ---- constants (X-Miner's own table) ----
def _consts():
    src = open(os.path.join(ROOT, "src", "algo", "quantus", "quantus_consts.hpp")).read()
    def arr(name):
        body = re.search(name + r"[^=]*=\s*\{(.*?)\};", src, re.S).group(1)
        return [int(x, 16) for x in re.findall(r"0x([0-9a-f]+)ull", body)]
    ini, inter, diag, term = arr("kInitialRc"), arr("kInternalRc"), arr("kMatrixDiag"), arr("kTerminalRc")
    return [ini[12 * r:12 * r + 12] for r in range(4)], inter, diag, [term[12 * r:12 * r + 12] for r in range(4)]
INIT, INTERNAL, DIAG, TERM = _consts()
DIAG1 = [(d << 32) % P for d in DIAG]

# ---- Python reference (src/algo/quantus/quantus.cpp permute) ----
def ref_external(s):
    s = list(s)
    for o in range(0, 12, 4):
        x = s[o:o + 4]
        t01, t23 = x[0] + x[1], x[2] + x[3]; t0123 = t01 + t23; t01123 = t0123 + x[1]; t01233 = t0123 + x[3]
        s[o:o + 4] = [(t01123 + t01) % P, (t01123 + 2 * x[2]) % P, (t01233 + t23) % P, (t01233 + 2 * x[0]) % P]
    sums = [(s[k] + s[k + 4] + s[k + 8]) % P for k in range(4)]
    return [(s[i] + sums[i % 4]) % P for i in range(12)]

# ---- constant table (kQgen in qshell.cu): INIT (48), INTERNAL (22), DIAG (12), DIAG1 (12), TERM (48) ----
K_BASE = 0  # offset of kQgen in constant bank 3 (from the shell's symbol table)
TABLE = [x for row in INIT for x in row] + INTERNAL + DIAG + DIAG1 + [x for row in TERM for x in row]
def kofs(kind, r=0, i=0):
    base = {"init": 0, "internal": 48, "diag": 70, "diag1": 82, "term": 94}[kind]
    idx = base + (r * 12 + i if kind in ("init", "term") else (r if kind == "internal" else i))
    return K_BASE + 8 * idx
CB = "0x3"  # constant bank of the table (__constant__ data)
def kc(off, half): return "c[%s][0x%x]" % (CB, off + 4 * half)

# ---- register map ----
S = [10 + 2 * k for k in range(12)]           # state element k in R{S[k]}:R{S[k]+1}
T = [34 + 2 * k for k in range(14)]           # temp pairs R34..R61

# ---- primitive ops ----


# 96-bit accumulators are (l, h, t) register triples (l, h an aligned pair when used as a WIDE addend).
def acc_addc(acc, off):        # acc += 64-bit table constant at off
    l, h, t = acc
    return [f"IADD3 R{l}, P0, R{l}, {kc(off, 0)}, RZ", f"IADD3.X R{h}, P1, R{h}, {kc(off, 1)}, RZ, P0, !PT",
            f"IMAD.X R{t}, RZ, RZ, R{t}, P1"]
def fold96(o, acc, tmp):       # o = (h:l) - t + t * 2^32, rare borrows uncorrected (as the CUDA fold96)
    l, h, t = acc
    return [f"IADD3 R{o}, P0, R{l}, -R{t}, RZ", f"IADD3.X R{tmp}, R{h}, -0x1, RZ, P0, !PT",
            f"IADD3 R{tmp}, P1, R{tmp}, R{t}, RZ", f"IMAD.X R{o+1}, RZ, RZ, R{tmp}, P1",
            f"IADD3 R{o}, R{o}, R{tmp}, -R{o+1}"]

# ---- rounds ----


def add_rc0(off):  # s0 += constant (lazy 64-bit add: a carry out of 2^64 is worth 2^32 - 1)
    x = S[0]
    return [f"IADD3 R{x}, P0, R{x}, {kc(off, 0)}, RZ", f"IADD3.X R{x+1}, P1, R{x+1}, {kc(off, 1)}, RZ, P0, !PT",
            f"IMAD.X R{x+1}, RZ, RZ, R{x+1}, P1", f"IADD3.X R{x}, R{x}, -0x1, RZ, !P1, !PT"]


# ==== pipelined version: multiplies split into head (WIDEs) / tail (reduction), two register + predicate sets ====
# set s: L, M, Q / E, F pairs, three singles, and the carry predicate that crosses head -> tail (P4 / P5).
def tset(s):
    if s == 0: return dict(L=T[4], M=T[5], Q=T[6], x=T[7], y=T[7] + 1, z=T[8], pc="P4")
    return dict(L=T[9], M=T[10], Q=T[11], x=T[12], y=T[12] + 1, z=T[8] + 1, pc="P5")

def mul3(s, o, a, b):
    """o = a * b. value = l0 + (l1 + m0 + c) 2^32 - (c + q1 + Pm + Pc), c = m1 + q0 (carry Pc).
    Head: 4 WIDE (predicate Pm). Tail: 8 ops on P0..P3 + Pm. ALU 20 / FMA 20 clk (cost model)."""
    r = tset(s); L, M, Q, c, A, B, Pm = r["L"], r["M"], r["Q"], r["x"], r["y"], r["z"], r["pc"]
    head = [f"IMAD.WIDE.U32 R{L}, R{a}, R{b}, RZ", f"IMAD.WIDE.U32 R{M}, R{a}, R{b+1}, RZ",
            f"IMAD.WIDE.U32 R{M}, {Pm}, R{a+1}, R{b}, R{M}", f"IMAD.WIDE.U32 R{Q}, R{a+1}, R{b+1}, RZ"]
    tail = [f"IADD3 R{c}, P0, R{M+1}, R{Q}, RZ",                    # c = m1 + q0
            f"IADD3 R{A}, P1, P2, R{L+1}, R{M}, R{c}",              # A = l1 + m0 + c, k = P1 + P2
            f"IADD3.X R{B}, P3, R{c}, R{Q+1}, RZ, {Pm}, P0",        # B = c + q1 + Pm + Pc, j = P3
            f"IMAD.X R{B}, RZ, RZ, R{B}, P1", f"IMAD.X R{B}, RZ, RZ, R{B}, P2",   # B += k (multiply pipe)
            f"IADD3.X R{A}, R{A}, RZ, RZ, P1, P2",                  # A += k (ALU)
            f"IADD3 R{o}, P1, R{L}, -R{B}, RZ",                     # lo = l0 - B, P1 = no borrow
            f"IADD3.X R{o+1}, R{A}, -0x2, RZ, P1, !P3"]             # hi = A - 1 - j + no-borrow
    return head, tail

def diag2(s, o, a, k, sig, sig2):
    """o = a * DIAG[k] + (sig pair + sig2 * 2^64). Head: WIDEs + carry folding into K on the FMA pipe (P6);
    tail on P0..P2 + Pf. value = e0 + (e1 + f0 + f1 + K) 2^32 - (f1 + K + Pf), K = Pe1 + Pe2 + sig2."""
    r = tset(s); E, F, A, B, K, Pf = r["L"], r["M"], r["x"], r["y"], r["z"], r["pc"]
    d0, d1 = kofs("diag", i=k), kofs("diag1", i=k)
    k1, k2 = f"IMAD.X R{K}, RZ, RZ, R{sig2}, P6", f"IMAD.X R{K}, RZ, RZ, R{K}, P6"
    head = [f"IMAD.WIDE.U32 R{E}, P6, R{a}, {kc(d0, 0)}, R{sig}", k1,
            f"IMAD.WIDE.U32 R{F}, R{a}, {kc(d0, 1)}, RZ",
            f"IMAD.WIDE.U32 R{E}, P6, R{a+1}, {kc(d1, 0)}, R{E}", k2,
            f"IMAD.WIDE.U32 R{F}, {Pf}, R{a+1}, {kc(d1, 1)}, R{F}"]
    tail = [f"IADD3 R{A}, P0, P1, R{E+1}, R{F}, R{F+1}",            # A = e1 + f0 + f1
            f"IADD3.X R{B}, R{F+1}, R{K}, RZ, {Pf}, !PT",           # B = f1 + K + Pf
            f"IADD3.X R{B}, R{B}, RZ, RZ, P0, P1",                  # B += k
            f"IADD3.X R{A}, R{A}, -0x1, R{K}, P0, P1",              # H = A - 1 + K + k
            f"IADD3 R{o}, P2, R{E}, -R{B}, RZ",                     # lo = e0 - B
            f"IADD3.X R{o+1}, R{A}, RZ, RZ, P2, !PT"]               # hi = H + no-borrow
    return head, tail

def interleave2(a, b):
    out = []
    for k in range(max(len(a), len(b))):
        if k < len(b): out.append(b[k])
        if k < len(a): out.append(a[k])
    return out

def _writes(ins):
    w = set()
    for t in ins:
        a = re.sub(r"^@!?P\d\s+", "", t).split(None, 1)[1].split(",")[0].strip()
        if a.startswith("R"): w.add(int(a[1:]))
    return w
def _reads(ins):
    r = set()
    for t in ins:
        args = re.sub(r"^@!?P\d\s+", "", t).split(None, 1)[1].split(",")
        for x in args[1:]: r |= {int(v) for v in re.findall(r"\bR(\d+)\b", x)}
        if "WIDE" in t:  # 64-bit addend reads the pair
            m = re.match(r"\s*R(\d+)", args[-1])
            if m: r.add(int(m.group(1)) + 1)
    return r

def pipeline(items):
    """items: ("mul", fn) where fn(set) -> (head, tail), or ("seq", instrs) which first completes the pending tail.
    Consecutive multiplies must be independent; each tail runs interleaved with the next head."""
    out, pending, s = [], None, 0
    for kind, x in items:
        if kind == "seq":
            if pending: out += pending; pending = None
            out += x
            continue
        head, tail = x(s); s ^= 1
        if pending and _writes(pending) & _reads(head):  # dependent on the pending result: finish it first
            out += pending; pending = None
        out += interleave2(pending, head) if pending else head
        pending = tail
    if pending: out += pending
    return out


def sbox_items(x):
    x2, x3 = T[0], T[1]
    return [("mul", lambda s: mul3(s, x2, x, x)), ("mul", lambda s: mul3(s, x3, x2, x)),
            ("mul", lambda s: mul3(s, x2, x2, x2)), ("mul", lambda s: mul3(s, x, x3, x2))]



# ==== three-input 96-bit adds (IADD3 with two carry-outs, IADD3.X with two carries in and out) ====
def acc3(dst, x, y, z):
    """dst = x + y + z; operands are 64-bit pairs (int) or 96-bit accs (tuple (l, h, t)); dst a 96-bit acc."""
    def parts(v): return (v, v + 1, None) if isinstance(v, int) else v
    (xl, xh, xt), (yl, yh, yt), (zl, zh, zt) = parts(x), parts(y), parts(z)
    dl, dh, dt = dst
    tops = [f"R{q}" for q in (xt, yt, zt) if q is not None] + ["RZ"] * 3
    return [f"IADD3 R{dl}, P0, P1, R{xl}, R{yl}, R{zl}",
            f"IADD3.X R{dh}, P2, P3, R{xh}, R{yh}, R{zh}, P0, P1",
            f"IADD3.X R{dt}, {tops[0]}, {tops[1]}, {tops[2]}, P2, P3"]
def acc2(dst, x, y):
    """dst = x + y (pairs or accs)."""
    def parts(v): return (v, v + 1, None) if isinstance(v, int) else v
    (xl, xh, xt), (yl, yh, yt) = parts(x), parts(y)
    dl, dh, dt = dst
    tops = [f"R{q}" for q in (xt, yt) if q is not None]
    top = (f"IMAD.X R{dt}, {tops[0]}, 0x1, {tops[1]}, P1" if len(tops) == 2 else
           f"IMAD.X R{dt}, RZ, RZ, {tops[0]}, P1" if len(tops) == 1 else f"IMAD.X R{dt}, RZ, RZ, RZ, P1")
    return [f"IADD3 R{dl}, P0, R{xl}, R{yl}, RZ", f"IADD3.X R{dh}, P1, R{xh}, R{yh}, RZ, P0, !PT", top]

def external3(rc=None):
    """out_g = M4 (x_g + z), z_k = x_k + x_{k+4} + x_{k+8} (+ rc[i] before the fold): z by 3-input adds; M4 as t01, s = t01+w2+w3, y0 = s+t01+w1, u = s+w2,
    v = s+w3, y1 = u+w1+w2, y2 = u+w3+w3, y3 = v+w0+w0 (8 adds instead of 12)."""
    out = []
    Z = [(T[k], T[k] + 1, T[4] + k if k < 2 else T[5] + k - 2) for k in range(4)]
    for k in range(4):
        out += acc3(Z[k], S[k], S[k + 4], S[k + 8])
    t01, sacc = (T[8], T[8] + 1, T[9]), (T[11], T[11] + 1, T[12])
    u, v = (T[10], T[10] + 1, T[9] + 1), (T[13], T[13] + 1, T[12] + 1)
    for g in range(3):
        W = [(S[4 * g + k], S[4 * g + k] + 1, T[6] + k if k < 2 else T[7] + k - 2) for k in range(4)]
        for k in range(4):
            out += acc2(W[k], S[4 * g + k], Z[k])
        w0, w1, w2, w3 = W
        out += acc2(t01, w0, w1) + acc3(sacc, t01, w2, w3) + acc3(t01, sacc, t01, w1)   # y0 in t01
        out += acc2(u, sacc, w2) + acc2(v, sacc, w3)
        out += acc3(w1, u, w1, w2)                                                      # y1 in w1
        out += acc3(w2, u, w3, w3)                                                      # y2 in w2
        out += acc3(w0, v, w0, w0)                                                      # y3 in w0
        # folds: y3 (w0 regs) -> S[3] (w3 pair) first, then y0 (t01) -> S[0] (w0 pair); y1, y2 in place
        dsts = [(w0, 3), (t01, 0), (w1, 1), (w2, 2)]
        for acc, k in dsts:
            if rc is not None: out += acc_addc(acc, rc[4 * g + k])
            out += fold96(S[4 * g + k], acc, w3[2] if k != 3 else u[0])
    return out

def sigma3():
    sig = (T[2], T[2] + 1, T[3])
    out = acc3(sig, S[0], S[1], S[2])
    for i in range(3, 11, 2): out += acc3(sig, sig, S[i], S[i + 1])
    out += acc2(sig, sig, S[11])
    return out
# ==== looped version (instruction-cache friendly): full rounds and partial rounds as loops, rc via LDC ====
# Extra table rows (appended, so earlier offsets stay valid): INITX = INIT[1..3] + zero row, TERMX = TERM[1..3] + zero.
ZROW = [0] * 12
TABLE += [x for row in INIT[1:] + [ZROW] for x in row] + [x for row in TERM[1:] + [ZROW] for x in row]
IDX = 8          # loop index register (outside the 26 pairs)
LOOPS = [0]      # label counter

def gadd_r(x, r):  # x += r (registers, lazy 64-bit; a carry out of 2^64 is worth 2^32 - 1)
    return [f"IADD3 R{x}, P0, R{x}, R{r}, RZ", f"IADD3.X R{x+1}, P1, R{x+1}, R{r+1}, RZ, P0, !PT",
            f"IMAD.X R{x+1}, RZ, RZ, R{x+1}, P1", f"IADD3.X R{x}, R{x}, -0x1, RZ, !P1, !PT"]

def external_rc_ldc(row_base):
    """external(None), then + rc[i] loaded with LDC from c[CB][IDX + row_base + 8 i] (all temps free again)."""
    out = external3(None)
    for i in range(12):
        out.append(f"LDC.64 R{T[i]}, c[{CB}][R{IDX}+0x{row_base + 8 * i:x}]")
    for i in range(12): out += gadd_r(S[i], T[i])
    return out

def full_round_body(row_base):
    return pipeline(_sbox_round_items()) + external_rc_ldc(row_base)

def _sbox_round_items():
    items = []
    X = [(T[0], T[1]), (T[2], T[3])]
    for e in range(0, 12, 2):
        pe = [S[e], S[e + 1]]
        for stage in range(4):
            for j in range(2):
                x, x2, x3 = pe[j], X[j][0], X[j][1]
                fn = [lambda s, x=x, x2=x2: mul3(s, x2, x, x), lambda s, x=x, x2=x2, x3=x3: mul3(s, x3, x2, x),
                      lambda s, x2=x2: mul3(s, x2, x2, x2), lambda s, x=x, x2=x2, x3=x3: mul3(s, x, x3, x2)][stage]
                items.append(("mul", fn))
    return items

def loop(body, count, step):
    """('loop', label, body, control): IDX runs 0, step, ..., (count-1) step; IDX is reset before the loop."""
    LOOPS[0] += 1
    lab = f".L_q_{LOOPS[0]}"
    ctl = [f"IADD3 R{IDX}, R{IDX}, 0x{step:x}, RZ", f"ISETP.NE.AND P6, PT, R{IDX}, 0x{count * step:x}, PT",
           f"@P6 BRA `({lab})"]
    return [("code", [f"MOV R{IDX}, RZ"]), ("loop", lab, body, ctl)]

def partial_loop_segments():
    """Prologue S_0; loop r = 0..20: Sigma_r, D0_r, s0 += INTERNAL[r+1] (LDC), S_{r+1} || D1..11_r; epilogue r = 21."""
    sig, sig2 = T[2], T[3]
    rcp = T[13]
    pro = pipeline(sbox_items(S[0]))
    items = [("seq", sigma3()), ("mul", lambda s: diag2(s, S[0], S[0], 0, sig, sig2)),
             ("seq", [f"LDC.64 R{rcp}, c[{CB}][R{IDX}+0x{kofs('internal', 1):x}]"] + gadd_r(S[0], rcp))]
    sb = sbox_items(S[0])
    ds = [("mul", lambda s, i=i: diag2(s, S[i], S[i], i, sig, sig2)) for i in range(1, 12)]
    for k in range(11):
        if k < 4: items.append(sb[k])
        items.append(ds[k])
    body = pipeline(items)
    epi_items = [("seq", sigma3())]
    sc = (T[13], T[13] + 1, T[3] + 1)
    for i in range(12):
        off = kofs("term", 0, i)
        epi_items.append(("seq", [f"IADD3 R{sc[0]}, P0, R{sig}, {kc(off, 0)}, RZ",
                                  f"IADD3.X R{sc[1]}, P1, R{sig+1}, {kc(off, 1)}, RZ, P0, !PT",
                                  f"IADD3.X R{sc[2]}, R{sig2}, RZ, RZ, P1, !PT"]))
        h, tl = diag2(0, S[i], S[i], i, sc[0], sc[2])
        epi_items.append(("seq", h + tl))
    return [("code", pro)] + loop(body, 21, 8) + [("code", pipeline(epi_items))]

def permutation_loops():
    """Segments: ('code', instrs) / ('loop', label, body, control)."""
    initx, termx = K_BASE + 8 * 142, K_BASE + 8 * 190
    segs = [("code", external3([kofs("init", 0, i) for i in range(12)]))]
    segs += loop(full_round_body(initx), 4, 96)
    segs += [("code", add_rc0(kofs("internal", 0)))]
    segs += partial_loop_segments()
    segs += loop(full_round_body(termx), 4, 96)
    return segs


def emit_segments(segs, list_sched=True):
    """Final ctrl-annotated lines: each straight block / loop body list-scheduled and stall-scheduled (loop bodies
    with wrap-around); LDC consumers wait on scoreboard 5."""
    lines = []
    def pad_last():  # latency state is not carried across segments: cover every fixed latency at the boundary
        for k in range(len(lines) - 1, -1, -1):
            if ":S" in lines[k] and "BRA" not in lines[k]:
                lines[k] = re.sub(r":S(\d\d)\]", lambda m: ":S%02d]" % max(7, int(m.group(1))), lines[k]); return
            if "BRA" in lines[k]: return  # loop exit: the ISETP -> BRA wait (13) already covers the body
    for sg in segs:
        pad_last()
        if sg[0] == "raw":  # control flow kept in order (labels / branches)
            lines += sass.schedule(sg[1], None, False)[0]
            continue
        if sg[0] == "code":
            body = sass.schedule_list(sg[1], window=400) if list_sched else sg[1]
            lines += sass.schedule(body, None, False)[0]
        else:
            _, lab, body, ctl = sg
            b = sass.schedule_list(body, window=400) if list_sched else body
            lines.append(f"  {lab}:")
            lines += sass.schedule(b + ctl)
    pad_last()
    return lines
# ==== miner block: state from (a, b) and job.pre, permutation 3 tail, absorb, permutation 4 -> s0 ====
PRE = 0x160  # job.pre[12] in the kernel parameters (QtcJob)
def gadd_pre(x, i, c):
    """S[i] = pre[i] + c (pair), lazy: a carry out of 2^64 is worth 2^32 - 1."""
    pl, ph = "c[0x0][0x%x]" % (PRE + 8 * i), "c[0x0][0x%x]" % (PRE + 8 * i + 4)
    return [f"IADD3 R{x}, P0, R{c}, {pl}, RZ", f"IADD3.X R{x+1}, P2, R{c+1}, {ph}, RZ, P0, !PT",
            f"IMAD.X R{x+1}, RZ, RZ, R{x+1}, P2", f"IADD3.X R{x}, R{x}, -0x1, RZ, !P2, !PT"]

def state_init(a, b):
    """c = (a + b, 3a + b, 2a + 3b, a + 2b) in T[0..3]; S[i] = pre[i] + c[i & 3] (x2 for elements 4..7)."""
    C = T[:4]
    out = [f"IADD3 R{C[0]}, P0, R{a}, R{b}, RZ", f"IADD3.X R{C[0]+1}, RZ, RZ, RZ, P0, !PT",
           f"IMAD.WIDE.U32 R{C[1]}, R{a}, 0x3, RZ", f"IADD3 R{C[1]}, P0, R{C[1]}, R{b}, RZ",
           f"IADD3.X R{C[1]+1}, R{C[1]+1}, RZ, RZ, P0, !PT",
           f"IMAD.WIDE.U32 R{C[2]}, R{b}, 0x3, RZ", f"IADD3 R{C[2]}, P0, P1, R{C[2]}, R{a}, R{a}",
           f"IADD3.X R{C[2]+1}, R{C[2]+1}, RZ, RZ, P0, P1",
           f"IADD3 R{C[3]}, P0, P1, R{a}, R{b}, R{b}", f"IADD3.X R{C[3]+1}, RZ, RZ, RZ, P0, P1"]
    D = T[4:8]  # 2c for elements 4..7
    for k in range(4):
        out += [f"IADD3 R{D[k]}, P0, R{C[k]}, R{C[k]}, RZ", f"IADD3.X R{D[k]+1}, R{C[k]+1}, R{C[k]+1}, RZ, P0, !PT"]
    for i in range(12): out += gadd_pre(S[i], i, D[i & 3] if 4 <= i < 8 else C[i & 3])
    return out

def tail_segments():
    return permutation_loops()[1:]


PASS = None  # pass counter register for the shared tail (set by splice)
def miner_segments_shared(a, b):
    """state; .pass: tail; if second pass -> done; absorb + external(INIT[0]); pass = 1; BRA .pass; .done"""
    ini = [kofs("init", 0, i) for i in range(12)]
    absorb = [f"IADD3 R{S[0]}, P0, R{S[0]}, 0x1, RZ", f"IADD3.X R{S[0]+1}, P1, R{S[0]+1}, RZ, RZ, P0, !PT",
              f"IMAD.X R{S[0]+1}, RZ, RZ, R{S[0]+1}, P1", f"IADD3.X R{S[0]}, R{S[0]}, -0x1, RZ, !P1, !PT",
              f"IADD3 R{S[1]}, P0, R{S[1]}, 0x1, RZ", f"IADD3.X R{S[1]+1}, P1, R{S[1]+1}, RZ, RZ, P0, !PT",
              f"IMAD.X R{S[1]+1}, RZ, RZ, R{S[1]+1}, P1", f"IADD3.X R{S[1]}, R{S[1]}, -0x1, RZ, !P1, !PT"]
    LOOPS[0] += 1; lp, ld = f".L_q_{LOOPS[0]}p", f".L_q_{LOOPS[0]}d"
    return ([("code", state_init(a, b) + [f"MOV R{PASS}, RZ"]), ("raw", [f"  {lp}:"])] + tail_segments() +
            [("raw", [f"ISETP.NE.AND P6, PT, R{PASS}, RZ, PT", f"@P6 BRA `({ld})"]),
             ("code", absorb + external3(ini) + [f"MOV R{PASS}, 0x1"]),
             ("raw", [f"BRA `({lp})", f"  {ld}:"])])

def ref_tail(s):
    s = list(s)
    for r in range(4):
        s = ref_external([pow(x, 7, P) for x in s])
        s = [(x + c) % P for x, c in zip(s, INIT[r + 1])] if r < 3 else [(s[0] + INTERNAL[0]) % P] + s[1:]
    for r in range(22):
        s[0] = pow(s[0], 7, P); t = sum(s) % P
        s = [(t + s[i] * DIAG[i]) % P for i in range(12)]
        s = [(s[0] + INTERNAL[r + 1]) % P] + s[1:] if r < 21 else [(x + c) % P for x, c in zip(s, TERM[0])]
    for r in range(4):
        s = ref_external([pow(x, 7, P) for x in s])
        if r < 3: s = [(x + c) % P for x, c in zip(s, TERM[r + 1])]
    return s

def ref_miner(pre, a, b):
    c = [a + b, 3 * a + b, 2 * a + 3 * b, a + 2 * b]
    s = [(pre[i] + (2 if 4 <= i < 8 else 1) * c[i & 3]) % P for i in range(12)]
    s = ref_tail(s)
    s[0] = (s[0] + 1) % P; s[1] = (s[1] + 1) % P
    s = [(x + c) % P for x, c in zip(ref_external(s), INIT[0])]
    return ref_tail(s)[0]

# ==== build: nvcc compiles qshell.cu, CuAssembler turns it into text, the generated block replaces the markers,
# the emulator checks the result against the Python reference, CuAssembler assembles it ====
# sha256 of DefaultInsAsmRepos.sm_86.txt in CuAssembler @96a9f72 after cuassembler_sm86.patch (encodings it lacks:
# IADD3 with two carry-outs, IADD3.X with two carries in and out, negated predicate operands)
REPOS_SHA256 = "5445e5c15780f6db09f2d9f4a18961916b79ce7994b5a822d359d5408037fb58"

def table_header():
    vals = ", ".join("0x%016xull" % v for v in TABLE)
    return ("// Generated by tools/qgen/qgen.py: INIT, INTERNAL, DIAG, DIAG*2^32 mod p, TERM, INIT[1..3]+0, TERM[1..3]+0\n"
            "__constant__ unsigned long long kQgen[%d] = {%s};\n" % (len(TABLE), vals))

INS = re.compile(r"\s*\[([^\]]*)\]\s+/\*[0-9a-f]+\*/\s+(.*?);")
def splice(src, regcount=64):
    """src: CuAssembler text of the shell (lines). Returns (new text, generated lines, (a, b, out0, out1))."""
    global S, T, IDX, PASS, K_BASE
    mk, idx = {}, []
    for i, l in enumerate(src):
        m = re.search(r"LOP3\.LUT (R\d+), (R\d+), 0xf1f4f0([01]),", l)
        if m: mk[int(m.group(3))] = (int(m.group(1)[1:]), int(m.group(2)[1:])); idx.append(i)
    assert len(idx) == 2, "marker XORs not found in the shell"
    # registers live across the block: read after it before being written
    live, written = set(), set()
    for l in src[idx[-1] + 1:]:
        m = INS.match(l)
        if not m: continue
        _, dr, sr, _, _ = sass.parse(m.group(2))
        live |= {r for r in sr if r not in written}
        written |= dr
    outs = {mk[0][0], mk[1][0]}
    live -= outs
    reserved = live | {1, mk[0][1], mk[1][1]} | outs
    free = [r for r in range(0, regcount - 2) if r not in reserved]  # R62/R63 fault at a 64-register count
    pairs = [r for r in free if r % 2 == 0 and r + 1 in free]
    assert len(pairs) >= 26, (len(pairs), sorted(reserved))
    used = {q for p in pairs[:26] for q in (p, p + 1)}
    singles = [r for r in free if r not in used]
    S, T, IDX, PASS = pairs[:12], pairs[12:26], singles[0], singles[1]
    sym = re.search(r"Symbol\[\d+\] \"kQgen\": Container\(.*?'st_value': (\d+)", "\n".join(src))
    assert sym, "kQgen not found in the shell"
    K_BASE = int(sym.group(1))
    a, b = mk[0][1], mk[1][1]
    segs = miner_segments_shared(a, b) + [("code", [f"MOV R{mk[0][0]}, R{S[0]}", f"MOV R{mk[1][0]}, R{S[0]+1}"])]
    lines = emit_segments(segs)
    src = list(src)
    for k in range(idx[0] - 1, -1, -1):  # latency into the block: the instruction before the markers stalls >= 6
        if INS.match(src[k]):
            src[k] = re.sub(r":S(\d\d)\]", lambda m: ":S%02d]" % max(6, int(m.group(1))), src[k]); break
    new = src[:idx[0]] + lines + [l for k, l in enumerate(src[idx[0]:], idx[0]) if k not in idx]
    txt = "\n".join(new)
    txt = re.sub(r"(EIATTR_REGCOUNT.*?\.word\tindex@\(xm_quantus_search\)\s*\n\s*/\*[0-9a-f]+\*/\s*\.word\t)0x[0-9a-f]+",
                 lambda m: m.group(1) + "0x%08x" % regcount, txt, flags=re.S)
    txt = re.sub(r"(\.__section_info\s+0x)[0-9a-f]{2}(00000[0-9a-f])", lambda m: m.group(1) + "%02x" % regcount + m.group(2), txt)
    txt = re.sub(r"SHI_REGISTERS=\d+", "SHI_REGISTERS=%d" % regcount, txt)
    return txt, lines, (a, b, mk[0][0], mk[1][0])

def verify(lines, regs, n, seed=1):
    """Emulates the generated block (its final instruction order, loops and branches) for n (pre, nonce) inputs,
    the first three edge cases; returns how many give an s0 different from the reference."""
    import emu
    a, b, o0, o1 = regs
    rng = random.Random(seed)
    bad = 0
    for k in range(n):
        pre = [[0] * 12, [P - 1] * 12, list(range(12))][k] if k < 3 else [rng.getrandbits(64) % P for _ in range(12)]
        na, nb = rng.getrandbits(32), rng.getrandbits(32)
        const = {}
        for j, v in enumerate(TABLE):
            const[kc(K_BASE + 8 * j, 0)] = v & M32; const[kc(K_BASE + 8 * j, 1)] = v >> 32
        for i, v in enumerate(pre):
            const["c[0x0][0x%x]" % (PRE + 8 * i)] = v & M32; const["c[0x0][0x%x]" % (PRE + 8 * i + 4)] = v >> 32
        e = emu.Emu(const)
        e.r[a], e.r[b] = na, nb
        e.run_program(lines)
        bad += (e.r[o0] | (e.r[o1] << 32)) % P != ref_miner(pre, na, nb)
    return bad

def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit("failed: %s\n%s%s" % (" ".join(cmd), r.stdout, r.stderr))
    return r

def main():
    ap = argparse.ArgumentParser(description="Build kernels/prebuilt/quantus_sm_86.cubin (generated SASS permutations).")
    ap.add_argument("--cuassembler", required=True, help="CuAssembler checkout with cuassembler_sm86.patch applied")
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--ccbin", default=None, help="host compiler directory for nvcc (Windows: cl.exe)")
    ap.add_argument("--python", default=sys.executable, help="interpreter with CuAssembler's requirements")
    ap.add_argument("--out", default=os.path.join(ROOT, "kernels", "prebuilt", "quantus_sm_86.cubin"))
    ap.add_argument("--check", type=int, default=64, help="emulated (pre, nonce) checks before assembling")
    ap.add_argument("--keep", help="directory to keep the intermediate files in")
    args = ap.parse_args()
    repos = os.path.join(args.cuassembler, "CuAsm", "InsAsmRepos", "DefaultInsAsmRepos.sm_86.txt")
    if hashlib.sha256(open(repos, "rb").read()).hexdigest() != REPOS_SHA256:
        sys.exit(f"{repos} is not the patched version: run 'git apply {os.path.join(HERE, 'cuassembler_sm86.patch')}' "
                 "in the CuAssembler checkout (commit 96a9f72)")
    work = args.keep or tempfile.mkdtemp(prefix="qgen")
    os.makedirs(work, exist_ok=True)
    tools = os.path.dirname(shutil.which(args.nvcc) or args.nvcc)
    env = dict(os.environ, PYTHONPATH=args.cuassembler, PATH=tools + os.pathsep + os.environ.get("PATH", ""))
    open(os.path.join(work, "qgen_table.h"), "w").write(table_header())
    shell, shell_txt = os.path.join(work, "qshell.cubin"), os.path.join(work, "qshell.cuasm")
    cmd = [args.nvcc, "-cubin", "-arch=sm_86", "-O3", "-std=c++17", "-I", work, "-o", shell, os.path.join(HERE, "qshell.cu")]
    if args.ccbin: cmd[1:1] = ["-ccbin", args.ccbin]
    run(cmd)
    cuasm = os.path.join(args.cuassembler, "bin", "cuasm.py")
    run([args.python, cuasm, shell, "-o", shell_txt], env)
    txt, lines, regs = splice(open(shell_txt).read().splitlines())
    bad = verify(lines, regs, args.check)
    print(f"generated block: {sum(1 for l in lines if '[B' in l)} instructions; emulator {args.check - bad}/{args.check} match")
    if bad: sys.exit("emulator check failed: cubin not written")
    gen_txt, gen = os.path.join(work, "generated.cuasm"), os.path.join(work, "generated.cubin")
    open(gen_txt, "w").write(txt)
    r = run([args.python, cuasm, gen_txt, "-o", gen], env)
    if not os.path.exists(gen) or "ERROR" in r.stdout + r.stderr:
        sys.exit("assembly failed:\n" + "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "rror" in l)[:2000])
    run([sys.executable, os.path.join(HERE, "cuasm_fix.py"), shell, gen], env)
    shutil.copyfile(gen, args.out)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
