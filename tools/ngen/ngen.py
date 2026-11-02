# Nexa (secp256k1) comb multiplications as generated sm_86 SASS: the build of kernels/prebuilt/nexa_sm_86.cubin.
# nvcc compiles nshell.cu (kernels/nexa/nexa.cu in its three-kernel form); in xm_nexa_b the marker XORs become this
# generator's block, which computes P = x G and R = k G with the signed 20-bit comb (13 windows, table entries of
# 64 bytes: affine x then y, 8 little-endian words each) and stores them to the nonce's pr row. The field arithmetic
# is field.py (8 x 32-bit words); nsass.py schedules; the emulator (nemu.py) checks the block against Python
# elliptic-curve arithmetic before anything is assembled. usage: ngen.py --cuassembler <dir> [--ccbin <dir>]
#
# Accepted slips, as in the CUDA kernel (they lose that nonce only; the CPU re-checks every candidate): a zero digit
# (probability 2^-19 per window) adds the entry for 1 instead of nothing; field ops drop a final carry with
# probability below 2^-200; the equal / opposite point cases of the addition are not handled.
import argparse, hashlib, os, random, re, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "qgen")]
import field, nemu, nsass, splice

W, NW = 20, 13
P = 2 ** 256 - 2 ** 32 - 977
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
TAB = "c[0x0][0x160]"  # xm_nexa_b's first parameter (the table pointer)
M32 = 0xFFFFFFFF


# ---- Python reference ----
def jdouble(p):
    X, Y, Z = p
    if Y == 0: return (0, 1, 0)
    S = 4 * X * Y * Y % P; M = 3 * X * X % P
    X3 = (M * M - 2 * S) % P
    return (X3, (M * (S - X3) - 8 * Y ** 4) % P, 2 * Y * Z % P)


def jadd(p, q):
    if p[2] == 0: return q
    if q[2] == 0: return p
    X1, Y1, Z1 = p; X2, Y2, Z2 = q
    U1, U2 = X1 * Z2 * Z2 % P, X2 * Z1 * Z1 % P
    S1, S2 = Y1 * Z2 ** 3 % P, Y2 * Z1 ** 3 % P
    if U1 == U2: return jdouble(p) if S1 == S2 else (0, 1, 0)
    H, R = (U2 - U1) % P, (S2 - S1) % P
    X3 = (R * R - H ** 3 - 2 * U1 * H * H) % P
    return (X3, (R * (U1 * H * H - X3) - S1 * H ** 3) % P, H * Z1 * Z2 % P)


def smul(k, pt=(GX, GY, 1)):
    acc, add = (0, 1, 0), pt
    while k:
        if k & 1: acc = jadd(acc, add)
        add = jdouble(add); k >>= 1
    return acc


def affine(p):
    zi = pow(p[2], P - 2, P)
    return p[0] * zi * zi % P, p[1] * zi ** 3 % P


class Table(dict):
    """The comb table as emulator memory, entries computed on first read: entry (i, m) at byte offset
    64 (i 2^(W-1) + m - 1) = m 2^(W i) G."""
    def __init__(self, base):
        super().__init__(); self.base = base; self.pts = {}

    def get(self, addr, default=0):
        if addr in self: return self[addr]
        off = addr - self.base
        if not 0 <= off < 64 * NW * 2 ** (W - 1): return default
        e, w = divmod(off, 64)
        i, m = divmod(e, 2 ** (W - 1))
        if e not in self.pts: self.pts[e] = affine(smul((m + 1) << (W * i)))
        v = self.pts[e][0] if w < 32 else self.pts[e][1]
        return v >> (8 * (w % 32)) & M32


# ---- register allocation ----
class Regs:
    def __init__(self, free):
        self.free = sorted(free)

    def take(self, n=1, align=1):
        for r in self.free:
            if r % align == 0 and all(r + q in self.free for q in range(n)):
                for q in range(n): self.free.remove(r + q)
                return r
        raise AssertionError("out of registers (%d x %d aligned %d; %d left)" % (1, n, align, len(self.free)))

    def fe(self):
        return [self.take() for _ in range(8)]


# ---- the block ----
def comb_block(vals, free):
    """Instruction segments: ('code', [...]) / ('label', name) / ('loop', label, body, ctl)."""
    xv, kv, rowv = vals[0:8], vals[8:16], vals[16:18]
    g = Regs(free)
    # aligned groups first
    s1 = [g.take(4, 4)]; s1.append(g.take(4, 4)); S1 = [s1[0] + q for q in range(4)] + [s1[1] + q for q in range(4)]
    s2 = [g.take(4, 4)]; s2.append(g.take(4, 4)); S2 = [s2[0] + q for q in range(4)] + [s2[1] + q for q in range(4)]
    tp = [g.take(2, 2) for _ in range(8)]
    T = [r + q for r in tp for q in (0, 1)]
    acc = [g.take(2, 2) for _ in range(2)]
    TB, A, ROW = g.take(2, 2), g.take(2, 2), g.take(2, 2)
    ov = [g.take(), g.take()]
    X, Y, Z, S0, S3, KS = g.fe(), g.fe(), g.fe(), g.fe(), g.fe(), g.fe()
    CARRY, WB = g.take(), g.take()
    sc = field.Scratch(list(T), list(acc), list(ov))
    KP0, KP1, KM = acc[0], acc[0] + 1, acc[1]  # p's words for the negation (the accumulators are free then)
    D, ND, MAG, OFF = T[0], T[1], T[2], T[3]
    pr = field.Preds(("P0", "P1", "P2", "P3", "P4"))

    def mul(o, a, b):
        sc.t = list(T); field.mul(body, pr, o, a, b, sc)

    def sqr(o, a):
        sc.t = list(T); field.sqr(body, pr, o, a, sc)

    def add(o, a, b): field.add(body, pr, o, a, b, T[0])

    def sub(o, a, b): field.sub(body, pr, o, a, b, T[0], T[1])

    def window(out):
        """Digit of this window (signed, with the carry), table entry address, loads into S1 (x) / S2 (y), y negated
        for a negative digit."""
        out += ["LOP3.LUT R%d, R%d, 0xfffff, RZ, 0xc0, !PT" % (D, KS[0])]
        out += ["SHF.R.U64 R%d, R%d, 0x14, R%d" % (KS[m], KS[m], KS[m + 1]) for m in range(7)]
        out += ["SHF.R.U32.HI R%d, RZ, 0x14, R%d" % (KS[7], KS[7]),
                "IADD3 R%d, R%d, R%d, RZ" % (D, D, CARRY),
                "ISETP.GT.U32.AND P5, PT, R%d, 0x80000, PT" % D,
                "IADD3 R%d, -R%d, 0x100000, RZ" % (ND, D),
                "SEL R%d, R%d, R%d, P5" % (MAG, ND, D),
                "SEL R%d, RZ, 0x1, !P5" % CARRY,
                "ISETP.EQ.U32.AND P4, PT, R%d, RZ, PT" % MAG,
                "SEL R%d, R%d, 0x1, !P4" % (MAG, MAG),
                "IMAD R%d, R%d, 0x40, R%d" % (OFF, MAG, WB),
                "IADD3 R%d, P0, R%d, R%d, RZ" % (A, TB, OFF),
                "IADD3.X R%d, R%d, RZ, RZ, P0, !PT" % (A + 1, TB + 1),
                "IADD3 R%d, R%d, 0x2000000, RZ" % (WB, WB)]
        for q, (dst, o) in enumerate([(s1[0], 0), (s1[1], 0x10), (s2[0], 0x20), (s2[1], 0x30)]):
            out.append("LDG.E.128.CONSTANT R%d, desc[UR4][R%d.64%s]" % (dst, A, "+0x%x" % o if o else ""))
        out += ["IMAD.MOV.U32 R%d, RZ, RZ, -0x3d1" % KP0, "IMAD.MOV.U32 R%d, RZ, RZ, -0x2" % KP1,
                "IMAD.MOV.U32 R%d, RZ, RZ, -0x1" % KM]
        Q = T[0:8]
        out.append("IADD3 R%d, P0, -R%d, R%d, RZ" % (Q[0], S2[0], KP0))
        prev = "P0"
        for j in range(1, 8):
            pj = "P%d" % (j % 4)
            out.append("IADD3.X R%d, %s, ~R%d, R%d, RZ, %s, !PT" % (Q[j], pj, S2[j], KP1 if j == 1 else KM, prev))
            prev = pj
        out += ["SEL R%d, R%d, R%d, P5" % (S2[j], Q[j], S2[j]) for j in range(8)]

    def madd():
        """(X, Y, Z) += (S1, S2) affine (madd-2007-bl as the CUDA kernel's madd), temporaries S0..S3."""
        sqr(S0, Z)           # Z1Z1
        mul(S1, S1, S0)      # U2
        mul(S2, S2, Z)
        mul(S2, S2, S0)      # S2
        sub(S1, S1, X)       # H
        sub(S2, S2, Y)
        add(S2, S2, S2)      # r
        sqr(S0, S1)          # HH
        add(S0, S0, S0)      # 2 HH
        mul(S3, S1, S0)
        add(S3, S3, S3)      # J
        mul(S0, X, S0)
        add(S0, S0, S0)      # V
        mul(Z, Z, S1)
        add(Z, Z, Z)         # Z3
        sqr(S1, S2)
        sub(S1, S1, S3)
        sub(S1, S1, S0)
        sub(X, S1, S0)       # X3
        sub(S1, S0, X)
        mul(S1, S2, S1)
        mul(S3, Y, S3)
        add(S3, S3, S3)
        sub(Y, S1, S3)       # Y3

    segs = []
    body = ["IMAD.MOV.U32 R%d, RZ, RZ, %s" % (TB, TAB), "IMAD.MOV.U32 R%d, RZ, RZ, c[0x0][0x%x]" % (TB + 1, int(TAB[-4:-1], 16) + 4),
            "IADD3 R%d, P0, R%d, -0x40, RZ" % (TB, TB), "IADD3.X R%d, R%d, -0x1, RZ, P0, !PT" % (TB + 1, TB + 1),
            "MOV R%d, R%d" % (ROW, rowv[0]), "MOV R%d, R%d" % (ROW + 1, rowv[1])]
    segs.append(("code", body))
    # pass loop: pass 0 = x (P), pass 1 = k (R); the pass is ROW == the original row address
    body = ["ISETP.EQ.U32.AND P6, PT, R%d, R%d, PT" % (ROW, rowv[0])]
    body += ["SEL R%d, R%d, R%d, P6" % (KS[i], xv[i], kv[i]) for i in range(8)]
    body += ["MOV R%d, RZ" % CARRY, "MOV R%d, RZ" % WB]
    window(body)
    body += ["MOV R%d, R%d" % (X[i], S1[i]) for i in range(8)]
    body += ["MOV R%d, R%d" % (Y[i], S2[i]) for i in range(8)]
    body += ["MOV R%d, %s" % (Z[i], "0x1" if i == 0 else "RZ") for i in range(8)]
    segs.append(("label", ".L_ng_pass"))
    segs.append(("code", body))
    body = []
    window(body)
    madd()
    segs.append(("loop", ".L_ng_win", body, ["ISETP.NE.AND P6, PT, R%d, 0x%x, PT" % (WB, NW << 25), "@P6 BRA `(.L_ng_win)"]))
    body = []
    for c, F in enumerate((X, Y, Z)):
        for i in range(8):
            body.append("STG.E desc[UR4][R%d.64+0x%x], R%d" % (ROW, 4 * (8 * c + i), F[i]))
    body += ["IADD3 R%d, P0, R%d, 0x60, RZ" % (ROW, ROW), "IADD3.X R%d, R%d, RZ, RZ, P0, !PT" % (ROW + 1, ROW + 1),
             "IADD3 R%d, R%d, 0xc0, RZ" % (T[0], rowv[0]),
             "ISETP.NE.AND P6, PT, R%d, R%d, PT" % (ROW, T[0]), "@P6 BRA `(.L_ng_pass)"]
    segs.append(("code", body))
    return segs


def emit(segs):
    """ctrl-annotated lines: straight code list-scheduled, loop bodies list-scheduled and wrap-scheduled; latency
    state is not carried across segment boundaries (the last instruction before one stalls >= 7)."""
    lines = []
    def pad():
        for k in range(len(lines) - 1, -1, -1):
            if ":S" in lines[k]:
                lines[k] = re.sub(r":S(\d\d)\]", lambda m: ":S%02d]" % max(7, int(m.group(1))), lines[k])
                if int(re.search(r":S(\d\d)\]", lines[k]).group(1)) > 8 and ":-:S" in lines[k]:
                    lines[k] = lines[k].replace(":-:S", ":Y:S")
                return
    for sg in segs:
        pad()
        if sg[0] == "label":
            lines.append("  %s:" % sg[1]); continue
        if sg[0] == "code":
            ctl = [t for t in sg[1] if "BRA" in t or ("ISETP" in t and "P6" in t and sg[1].index(t) >= len(sg[1]) - 2)]
            body = [t for t in sg[1] if t not in ctl]
            lines += nsass.schedule(nsass.schedule_list(body, window=600) + ctl, None, False)[0]
        else:
            _, lab, body, ctl = sg
            lines.append("  %s:" % lab)
            lines += nsass.schedule(nsass.schedule_list(body, window=600) + ctl)
    pad()
    # the pass loop's back edge waits for the stores' register reads (the next pass overwrites X, Y, Z)
    lines = [re.sub(r"\[B(.)(.)(.)-", lambda m: "[B%s%s%s3" % m.groups(), l, count=1) if "BRA `(.L_ng_pass)" in l else l
             for l in lines]
    return lines


def gen(vals, free):
    return emit(comb_block(vals, free))


# ---- emulator check ----
def check(lines, vals, n, seed=1):
    rng = random.Random(seed)
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    tab_base, row = 0x7000_0000_0000, 0x5000_0000_0000
    bad = 0
    for t in range(n):
        x = rng.randrange(1, N)  # (small scalars have zero digits: accepted slips)
        k = rng.randrange(1, N)
        mem = Table(tab_base)
        e = nemu.Emu({TAB: tab_base & M32, "c[0x0][0x%x]" % (int(TAB[-4:-1], 16) + 4): tab_base >> 32}, mem)
        for i in range(8): e.r[vals[i]] = x >> (32 * i) & M32; e.r[vals[8 + i]] = k >> (32 * i) & M32
        e.r[vals[16]], e.r[vals[17]] = row & M32, row >> 32
        e.run_program(lines)
        words = [mem.get(row + 4 * i, None) for i in range(48)]
        if None in words: bad += 1; continue
        def pt(o):
            c = [sum(words[o + 8 * j + i] << (32 * i) for i in range(8)) for j in range(3)]
            return c
        for s, o in ((x, 0), (k, 24)):
            Xj, Yj, Zj = pt(o)
            if Zj % P == 0 or affine((Xj % P, Yj % P, Zj % P)) != affine(smul(s)):
                bad += 1; print("mismatch: scalar %x (%s)" % (s, "P" if o == 0 else "R")); break
    return bad


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit("failed: %s\n%s%s" % (" ".join(cmd), r.stdout, r.stderr))
    return r


def main():
    ap = argparse.ArgumentParser(description="Build kernels/prebuilt/nexa_sm_86.cubin (generated comb code).")
    ap.add_argument("--cuassembler", required=True, help="CuAssembler checkout with tools/qgen/cuassembler_sm86.patch")
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--ccbin", default=None, help="host compiler directory for nvcc (Windows: cl.exe)")
    ap.add_argument("--python", default=sys.executable, help="interpreter with CuAssembler's requirements")
    ap.add_argument("--out", default=os.path.join(ROOT, "kernels", "prebuilt", "nexa_sm_86.cubin"))
    ap.add_argument("--check", type=int, default=6, help="emulated (x, k) checks before assembling")
    ap.add_argument("--keep", help="directory to keep the intermediate files in")
    args = ap.parse_args()
    work = args.keep or tempfile.mkdtemp(prefix="ngen")
    os.makedirs(work, exist_ok=True)
    tools = os.path.dirname(shutil.which(args.nvcc) or args.nvcc)
    env = dict(os.environ, PYTHONPATH=args.cuassembler, PATH=tools + os.pathsep + os.environ.get("PATH", ""))
    shell, shell_txt = os.path.join(work, "nshell.cubin"), os.path.join(work, "nshell.cuasm")
    cmd = [args.nvcc, "-cubin", "-arch=sm_86", "-O3", "-std=c++17", "-o", shell, os.path.join(HERE, "nshell.cu")]
    if args.ccbin: cmd[1:1] = ["-ccbin", args.ccbin]
    run(cmd)
    cuasm = os.path.join(args.cuassembler, "bin", "cuasm.py")
    run([args.python, cuasm, shell, "-o", shell_txt], env)
    txt, lines, vals, free = splice.splice(open(shell_txt).read().splitlines(), gen, "xm_nexa_b", 128)
    bad = check(lines, vals, args.check)
    print("generated block: %d instructions; emulator %d/%d match" % (sum(1 for l in lines if "[B" in l), args.check - bad, args.check))
    if bad: sys.exit("emulator check failed: cubin not written")
    gen_txt, gen_cubin = os.path.join(work, "generated.cuasm"), os.path.join(work, "generated.cubin")
    open(gen_txt, "w").write(txt)
    r = run([args.python, cuasm, gen_txt, "-o", gen_cubin], env)
    if not os.path.exists(gen_cubin) or "ERROR" in r.stdout + r.stderr:
        sys.exit("assembly failed:\n" + "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "rror" in l)[:3000])
    run([sys.executable, os.path.join(HERE, "..", "qgen", "cuasm_fix.py"), shell, gen_cubin], env)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shutil.copyfile(gen_cubin, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
