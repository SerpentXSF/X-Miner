# Splicing generated SASS into an nvcc-compiled shell (CuAssembler text), as tools/qgen does for Quantus, with any
# number of markers: the shell runs `xor.b32 r, r, 0x0f1f4fNN` (NN = 00, 01, ...) on the block's values and pins the
# results with stores right after (ptxas sinks pure instructions to their uses). ptxas still interleaves the
# markers with other work and reuses their destination registers, so:
#   - marker k becomes `MOV Vk, src` (its control code kept: it carries the waits for the loads feeding it), Vk a
#     register nothing else in the region touches;
#   - the generated block goes after the last marker and works in place on V0, V1, ...;
#   - instructions between the markers that (transitively) depend on a marker move after the block, their reads of
#     marker results renamed to the V registers;
#   - after the block, each register whose last definition in the region is a marker gets its V value back (MOV).
# Registers live after the block are reserved; the rest (below the register count) are free.
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qgen"))
import sass

INS = re.compile(r"(\s*\[[^\]]*\]\s+/\*[0-9a-f]+\*/\s+)(.*?)( ;.*)$")
MARK = re.compile(r"LOP3\.LUT (R\d+), (R\d+)(?:\.reuse)?, 0xf1f4f([0-9a-f]{2}), RZ, 0x3c, !PT")


def section(src, kernel):
    lo = next(i for i, l in enumerate(src) if re.match(r"\s*\.section\s+\.text\.%s," % re.escape(kernel), l))
    hi = next((i for i in range(lo + 1, len(src)) if re.match(r"\s*\.section\s", src[i])), len(src))
    return lo, hi


def _parse(text):
    """sass.parse plus the high halves of 64-bit addresses ([R2.64] reads R2 and R3)."""
    opc, dr, sr, dp, sp = sass.parse(text)
    sr = sr | {int(r) + 1 for r in re.findall(r"\bR(\d+)\.64", text)}
    return opc, dr, sr, dp, sp


def _ins(l):
    m = INS.match(l)
    return (m, _parse(m.group(2))) if m else (None, None)


def _rename_reads(l, ren):
    """Renames source registers of instruction line l by ren (dict reg -> reg); destinations stay."""
    m = INS.match(l)
    opc, rest = (m.group(2).split(None, 1) + [""])[:2]
    g = re.match(r"^(@!?P\d\s+)", m.group(2))
    if g: opc, rest = (m.group(2)[g.end():].split(None, 1) + [""])[:2]
    args = [a.strip() for a in rest.split(",")] if rest else []
    _, dr, sr, _, _ = sass.parse(m.group(2))
    k = 0
    if args and re.match(r"R\d+$", args[0]) and not opc.startswith(("STG", "STS", "ST", "RED", "ATOM")):
        k = 1  # destination
    for j in range(k, len(args)):
        new = re.sub(r"\bR(\d+)\b", lambda q: "R%d" % ren.get(int(q.group(1)), int(q.group(1))), args[j])
        args[j] = new.replace(".reuse", "") if new != args[j] else new  # (operand reuse flags: renamed ones dropped)
    txt = (g.group(1) if g else "") + opc + (" " + ", ".join(args) if args else "")
    return m.group(1) + txt + m.group(3)


def splice(src, gen, kernel, regcount, nvals=None):
    """src: shell lines. gen(vals, free) -> ctrl-annotated lines working in place on vals (one register per marker,
    by marker index), using only `free`. Returns (new text, generated lines, vals, free)."""
    lo, hi = section(src, kernel)
    mk, idx = {}, []
    for i in range(lo, hi):
        m = MARK.search(src[i])
        if m: mk[int(m.group(3), 16)] = (int(m.group(1)[1:]), int(m.group(2)[1:])); idx.append(i)
    assert mk and sorted(mk) == list(range(len(mk))), "markers missing or dropped: %s" % sorted(mk)
    first, last = idx[0], idx[-1]
    # pass 1: classify the region's other instructions (taint = depends on a marker result -> moves after the block)
    kind, taint = {}, set()
    for i in range(first, last + 1):
        if i in idx:
            taint.add(int(MARK.search(src[i]).group(1)[1:])); continue
        mi, info = _ins(src[i])
        if mi:
            moved_ = bool((info[1] | info[2]) & taint)
            kind[i] = "moved" if moved_ else "kept"
            if moved_: taint |= info[1]
    after = src[last + 1:hi]
    # registers V may not take: anything the region's other instructions touch, the markers' sources, and what is
    # read after the region before being written
    avoid = {1}
    for i, k in kind.items():
        _, info = _ins(src[i])
        avoid |= info[1] | info[2]
    # ... and anything the code before the region touches: a load there may still be reading its address registers
    # (read scoreboard) when the markers' MOVs run
    for l in src[lo:first]:
        mi, info = _ins(l)
        if mi: avoid |= info[1] | info[2]
    for i in idx:
        avoid.add(int(MARK.search(src[i]).group(2)[1:]))
    live, written = set(), set()
    for l in after:
        mi, info = _ins(l)
        if not mi: continue
        live |= {r for r in info[2] if r not in written}
        written |= info[1]
    # (marker destinations are rewritten after the block, so they are not live into it)
    cand = [r for r in range(0, regcount - 2) if r not in avoid and r not in live]
    # pass 2: markers -> MOV V, src; moved instructions with reads of marker results renamed (cur: reg -> V)
    before, moved, vals, cur = [], [], {}, {}
    for i in range(first, last + 1):
        l = src[i]
        if i in idx:
            m = MARK.search(l)
            k, d = int(m.group(3), 16), int(m.group(1)[1:])
            assert cand, "no register left for marker values"
            v = cand.pop(0)
            vals[k] = v
            before.append(l.replace(m.group(0), "MOV R%d, %s" % (v, m.group(2))).replace(".reuse", ""))
            cur[d] = v
            continue
        if i in kind:
            _, info = _ins(l)
            if kind[i] == "moved": moved.append(_rename_reads(l, cur))
            else: before.append(l)
            for r in info[1]: cur.pop(r, None)
            continue
        before.append(l)
    vals = [vals[k] for k in range(len(vals))]
    # after the block: the registers whose final definition in the region is a marker get their value back
    fin = ["        [B------:R-:W-:-:S01]         MOV R%d, R%d ;" % (d, v) for d, v in sorted(cur.items())]
    rest = fin + moved + after
    live, written = set(), set()
    for l in rest:
        mi, info = _ins(l)
        if not mi: continue
        live |= {r for r in info[2] if r not in written}
        written |= info[1]
    reserved = (live - set(vals)) | {1} | set(vals)
    free = [r for r in range(0, regcount - 2) if r not in reserved]
    lines = gen(vals, free)
    # memory scoreboards: the shell's loads / stores still in flight may read registers the block reuses (and the
    # block's may read registers the shell writes after it), so the block starts and ends waiting for all six
    lines = (["        [B012345:R-:W-:-:S01]         NOP ;"] + lines +
             ["        [B012345:R-:W-:-:S01]         NOP ;"])
    for k in range(len(before) - 1, -1, -1):  # latency into the block: the last instruction before it stalls >= 6
        if INS.match(before[k]):
            before[k] = re.sub(r":S(\d\d)\]", lambda m: ":S%02d]" % max(6, int(m.group(1))), before[k]); break
    for k in range(len(lines) - 1, -1, -1):  # and out of it: >= 7
        if ":S" in lines[k]:
            lines[k] = re.sub(r":S(\d\d)\]", lambda m: ":S%02d]" % max(7, int(m.group(1))), lines[k]); break
    new = src[:first] + before + lines + rest + src[hi:]
    s_lo, s_hi = section(new, kernel)
    sec = "\n".join(new[s_lo:s_hi])
    sec = re.sub(r"(\.__section_info\s+0x)[0-9a-f]{1,2}(00000[0-9a-f])", lambda m: m.group(1) + "%02x" % regcount + m.group(2), sec, count=1)
    sec = re.sub(r"SHI_REGISTERS=\d+", "SHI_REGISTERS=%d" % regcount, sec, count=1)
    txt = "\n".join(new[:s_lo]) + "\n" + sec + "\n" + "\n".join(new[s_hi:])
    k = re.escape(kernel)
    pat = r"(EIATTR_REGCOUNT(?:(?!EIATTR_)[\s\S])*?\.word\tindex@\(%s\)\s*\n\s*/\*[0-9a-f]+\*/\s*\.word\t)0x[0-9a-f]+" % k
    txt, n = re.subn(pat, lambda m: m.group(1) + "0x%08x" % regcount, txt)
    assert n == 1, "EIATTR_REGCOUNT of %s not found" % kernel
    return txt, lines, vals, free
