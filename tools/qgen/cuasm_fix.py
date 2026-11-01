# After CuAssembler: memory instructions (LDG/STG/LD/ST/ATOM/RED/LDC/...) whose text + control match an instruction of
# the original cubin get the original's exact encoding back (CuAssembler's sm_86 tables mis-set a bit of the
# desc[UR] forms: nvdisasm prints desc[URx] where cuobjdump and the encoding have none). usage: cuasm_fix.py <original.cubin> <assembled.cubin>   (patches the second in place)
import re, subprocess, sys

def dump(path):
    funcs, fn, pend = {}, None, None
    for l in subprocess.run(["cuobjdump", "-sass", path], capture_output=True, text=True).stdout.splitlines():
        m = re.search(r"Function : (\S+)", l)
        if m: fn = m.group(1); funcs[fn] = []; continue
        m = re.match(r"\s+/\*([0-9a-f]{4,5})\*/\s+(.*?;)\s+/\* 0x([0-9a-f]{16}) \*/", l)
        if m and fn: pend = (int(m.group(1), 16), re.sub(r"desc\[UR\d+\]", "", re.sub(r"\s+", " ", m.group(2))), int(m.group(3), 16)); continue
        m = re.match(r"\s+/\* 0x([0-9a-f]{16}) \*/", l)
        if m and pend and fn: funcs[fn].append(pend + (int(m.group(1), 16),)); pend = None
    return funcs

MEM = re.compile(r"^(@!?P\d )?(LDG|STG|LD|ST|ATOM|ATOMG|RED|LDS|STS|LDC|ULDC|LDGSTS)\b")
CTRL = 0xFFFFFE0000000000
orig, new = dump(sys.argv[1]), dump(sys.argv[2])
known = {}
for ins in orig.values():
    for (_, text, lo, hi) in ins:
        if MEM.match(text): known[(text, hi & CTRL)] = (lo, hi)
data = bytearray(open(sys.argv[2], "rb").read())
fixed = 0
for fn, ins in new.items():
    head = b"".join(lo.to_bytes(8, "little") + hi.to_bytes(8, "little") for (_, _, lo, hi) in ins[:6])
    pos = data.find(head)
    if pos < 0: print("function not found:", fn); continue
    for (a, text, lo, hi) in ins:
        k = (text, hi & CTRL)
        if MEM.match(text) and k in known and known[k] != (lo, hi):
            o = pos + a
            data[o:o + 8] = known[k][0].to_bytes(8, "little"); data[o + 8:o + 16] = known[k][1].to_bytes(8, "little")
            fixed += 1
# Loads/stores the original never had with the desc bit (hi bit 37): clear it where CuAssembler set it.
orig_desc = any((hi >> 37) & 1 for ins in orig.values() for (_, t, _, hi) in ins if MEM.match(t) and t.split()[0].startswith(("LDG", "STG")))
if not orig_desc:
    for fn, ins in new.items():
        head = b"".join(lo.to_bytes(8, "little") + hi.to_bytes(8, "little") for (_, _, lo, hi) in ins[:6])
        pos = data.find(head)
        for (a, text, lo, hi) in ins:
            if text.split()[0].startswith(("LDG", "STG")) and (hi >> 37) & 1:
                o = pos + a + 8
                data[o:o + 8] = (hi & ~(1 << 37)).to_bytes(8, "little"); fixed += 1
open(sys.argv[2], "wb").write(data)
print("fixed", fixed, "memory instructions")
