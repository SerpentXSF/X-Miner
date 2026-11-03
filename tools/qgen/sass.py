# sm_86 SASS scheduling for the Quantus generator: pipe-aware latencies (GA106, measured), a list scheduler that
# reorders straight-line code, and the stall/scoreboard pass that writes CuAssembler control codes.
import re

def pipe(text):
    op = re.sub(r"^@!?P\d\s+", "", text).split()[0]
    if "WIDE" in op: return "W"
    if op.startswith("IMAD.HI"): return "H"
    if op.startswith("IMAD"): return "F"
    if op.startswith("ISETP"): return "S"
    if op.startswith("LDC"): return "L"
    return "A"
def lat2(prod, cons, pred):
    """Producer -> consumer latency (ptxas schedule of the Quantus kernel, 2026-10-31): 4 within a pipe, 5 across
    ALU <-> multiply pipe, 6 WIDE -> WIDE and for a WIDE carry read on the multiply pipe; HI 13; ISETP -> branch 13."""
    if prod == "L": return 30  # LDC: variable latency (scoreboard); for ordering in the list scheduler only
    if prod == "H": return 13
    if prod == "S": return 13 if cons in ("B",) else 5
    if prod == "W":
        if cons == "W": return 6
        if pred and cons in ("F", "W"): return 6
        return 5
    same = (prod == "A") == (cons == "A")
    return 4 if same else 5


def schedule(body, ready=None, wrap=True):
    """body: list of instruction strings (no ctrl). Returns ctrl-annotated lines with stall counts covering latencies.
    wrap: the body is a loop; a second pass starts from the first pass's end state so loop-carried reads are covered."""
    if wrap:
        _, end_ready, end_cyc = schedule(body, None, False)
        lines, _, _ = schedule(body, {k: (v[0] - end_cyc, v[1]) for k, v in end_ready.items()}, False)
        return lines
    ready = dict(ready or {})  # reg/pred -> cycle when readable
    out, cyc = [], 0
    ldc_pending = set()  # registers written by LDC (scoreboard 5) not yet waited for
    ldc_reading = set()  # LDC source registers (read scoreboard 4): a later writer must wait
    for text in body:
        if text.rstrip().endswith(":"):  # label
            out.append([0, text, "", ""]); continue
        t = re.sub(r"^@!?P\d\s+", "", text)
        opc, rest = (t.split(None, 1) + [""])[:2]
        args = [a.strip() for a in rest.rstrip(" ;").split(",")] if rest else []
        dsts, srcs = [], []
        g = re.match(r"^@!?P(\d)\s", text)
        if g: srcs.append(("P", int(g.group(1))))
        if args:
            a0 = args[0]
            if opc.startswith("ISETP") and re.match(r"P\d$", a0):
                dsts.append(("P", int(a0[1:]))); srcargs = args[1:]
            elif re.match(r"R\d+$", a0):
                d = int(a0[1:]); dsts.append(("R", d))
                if "WIDE" in opc or opc.endswith(".64"): dsts.append(("R", d + 1))
                k = 1
                while k < len(args) and re.match(r"!?P\d$", args[k]) and args[k] != "PT":
                    dsts.append(("P", int(args[k][-1]))); k += 1
                srcargs = args[k:]
            else:
                srcargs = args
            for j, a in enumerate(srcargs):
                for r in [int(x) for x in re.findall(r"\bR(\d+)\b", a)]:
                    srcs.append(("R", r))
                    if "WIDE" in opc and j == len(srcargs) - 1 - sum(1 for x in srcargs if re.match(r"!?P\d$", x)): srcs.append(("R", r + 1))
                for p in re.findall(r"\bP(\d)\b", a): srcs.append(("P", int(p)))
        cons = "B" if opc == "BRA" else pipe(t)
        wait = ""
        if any(s[0] == "R" and s[1] in ldc_pending for s in srcs) or any(d[0] == "R" and d[1] in ldc_pending for d in dsts):
            wait += "5"; ldc_pending = set()
        if any(d[0] == "R" and d[1] in ldc_reading for d in dsts):
            wait += "4"; ldc_reading = set()
        need = max([ready[s][0] + lat2(ready[s][1], cons, s[0] == "P") if s in ready else 0 for s in srcs] + [cyc])
        prev = next((o for o in reversed(out) if o[1] and not o[1].rstrip().endswith(":")), None)
        if prev is None and need > cyc:  # loop-carried wait at the top of the body
            out.append([min(15, need - cyc), "NOP ;", "", ""]); cyc += out[-1][0]; prev = out[-1]
        if prev is not None and need > cyc:  # extend the previous instruction's stall
            extra = need - cyc
            add = min(15 - prev[0], extra)
            prev[0] += add; cyc += add
            while cyc < need:  # still short: insert NOPs
                out.append([min(15, need - cyc), "NOP ;", "", ""]); cyc += out[-1][0]
        out.append([1, text if text.endswith(";") else text + " ;", wait, "5" if opc.startswith("LDC") else ""])
        if opc.startswith("LDC"):
            for d in dsts: ldc_pending.add(d[1]); ready[d] = (cyc - 100, "A")
            for s in srcs:
                if s[0] == "R": ldc_reading.add(s[1])
        else:
            for d in dsts: ready[d] = (cyc, pipe(t))
        cyc += 1
    # ptxas pairs stalls above 8 with the yield flag (S13 without it is an invalid encoding)
    def ctrl(s, x, w, wb):
        if x.rstrip().endswith(":"): return x
        mask = "".join(str(k) if str(k) in w else "-" for k in range(6))
        rb = "4" if x.lstrip().startswith("LDC") else "-"
        return "        [B%s:R%s:W%s:%s:S%02d]         %s" % (mask, rb, wb or "-", "Y" if s > 8 else "-", s, x)
    lines = [ctrl(*o) for o in out]
    return lines, ready, cyc


def parse(text):
    """-> (opc, dst regs, src regs, dst preds, src preds)"""
    g = re.match(r"^@!?(P\d)\s+", text)
    t = re.sub(r"^@!?P\d\s+", "", text).rstrip(" ;")
    opc, rest = (t.split(None, 1) + [""])[:2]
    args = [a.strip() for a in rest.split(",")] if rest else []
    dr, sr, dp, sp = set(), set(), set(), set()
    if g: sp.add(g.group(1))
    if not args: return opc, dr, sr, dp, sp
    k = 0
    if opc.startswith("ISETP"):
        dp.add(args[0]); k = 1
    elif re.match(r"R\d+$", args[0]):
        d = int(args[0][1:]); dr.add(d)
        if "WIDE" in opc or opc.endswith(".64"): dr.add(d + 1)
        k = 1
        while k < len(args) and re.match(r"P\d$", args[k]): dp.add(args[k]); k += 1
    srcs = args[k:]
    for a in srcs:
        for r in re.findall(r"\bR(\d+)\b", a): sr.add(int(r))
        m = re.match(r"!?(P\d)$", a)
        if m: sp.add(m.group(1))
    if "WIDE" in opc and len(srcs) >= 3:
        m = re.match(r"-?R(\d+)", srcs[2])
        if m: sr.add(int(m.group(1)) + 1)
    if opc.startswith("STG") and srcs and ".64" in opc:
        m = re.match(r"R(\d+)$", srcs[-1])
        if m: sr.add(int(m.group(1)) + 1)
    return opc, dr, sr, dp, sp

REUSE_SCHED = 0
_SLOT_PRED = re.compile(r"^!?(P[0-6T]|UP[0-9])$")
_SLOT_OPS = ("IADD3", "IMAD", "LOP3", "SHF", "ISETP", "SEL")
def _src_slots(t):
    opc = t.split()[0]
    toks = [x.strip() for x in t[len(opc):].split(",")]
    return opc, [x.lstrip("-") for x in toks[1:] if not _SLOT_PRED.match(x)][:3]
def _share(a, b):
    """source slots (A, B, C) holding the same register in a and then b (operand reuse cache candidates)"""
    if a is None: return 0
    oa, sa = _src_slots(a); ob, sb = _src_slots(b)
    if not (oa.startswith(_SLOT_OPS) and ob.startswith(_SLOT_OPS)): return 0
    return sum(1 for k in range(min(len(sa), len(sb), 2 if "WIDE" in oa + ob else 3))
               if sa[k] == sb[k] and re.match(r"^R\d+$", sa[k]))

def schedule_list(prog, window=None):
    """List scheduler for straight-line code: dependency DAG over registers and predicates (RAW / WAR / WAW;
    predicates keep the names the generator chose, so independent chains must use different predicates to overlap).
    Greedy: issue the ready instruction that can start earliest (pipe-aware latencies), ties by critical path, then
    program order. schedule() afterwards sets the stall counts. Never deadlocks (no allocation)."""
    n = len(prog)
    info = [parse(t) for t in prog]
    preds = [[] for _ in range(n)]  # (j, kind)
    last_w, readers = {}, {}
    for i, (opc, dr, sr, dp, sp) in enumerate(info):
        srcs = [("R", r) for r in sr] + [("P", p) for p in sp]
        dsts = [("R", r) for r in dr] + [("P", p) for p in dp]
        for s in srcs:
            if s in last_w: preds[i].append((last_w[s], "praw" if s[0] == "P" else "raw"))
        for d in dsts:
            if d in last_w: preds[i].append((last_w[d], "war"))
            for j in readers.get(d, []):
                if j != i: preds[i].append((j, "war"))
        for s in srcs: readers.setdefault(s, []).append(i)
        for d in dsts: last_w[d] = i; readers[d] = []
    succ = [[] for _ in range(n)]
    npending = [0] * n
    for i in range(n):
        js = set(j for j, _ in preds[i])
        npending[i] = len(js)
        for j in js: succ[j].append(i)
    pipes = [pipe(t) for t in prog]
    prio = [0] * n
    for i in range(n - 1, -1, -1):
        prio[i] = 1 + max((prio[s] + 4 for s in succ[i]), default=0)
    done_t = [None] * n
    ready = [i for i in range(n) if npending[i] == 0]
    out, t, first = [], 0, 0
    sched = [False] * n
    while ready:
        while first < n and sched[first]: first += 1
        best, bk = None, None
        for i in ready:
            if window and i > first + window: continue
            est = t
            for j, k in preds[i]:
                est = max(est, done_t[j] + (0 if k == "war" else lat2(pipes[j], pipes[i], k == "praw")))
            if REUSE_SCHED:  # among instructions that can issue now, prefer one that reuses an operand slot
                key = (max(est, t), -_share(out[-1] if out else None, prog[i]), -prio[i], i)
            else:
                key = (est, -prio[i], i)
            if best is None or key < bk: best, bk = i, key
        i = best; ready.remove(i); sched[i] = True
        t = max(t, bk[0]); done_t[i] = t; t += 1
        out.append(prog[i])
        for s in succ[i]:
            npending[s] -= 1
            if npending[s] == 0: ready.append(s)
    assert len(out) == n
    return out
