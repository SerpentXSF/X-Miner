# Emulator for the SASS subset the Nexa generator emits (one thread): qgen's emu plus bitwise-not operands (~R),
# negated predicate operands, funnel shifts, SEL, LOP3, MOV and IMAD.MOV, ISETP compares, LDG from a dict.
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qgen"))
import emu
M32 = 0xFFFFFFFF


class Emu(emu.Emu):
    def __init__(self, const=None, mem=None):
        super().__init__(const or {})
        self.mem = mem if mem is not None else {}  # byte address -> u32 (4-aligned)

    def rd(self, tok):
        t = tok.replace(".reuse", "").strip()
        if t.startswith("~"):
            return (~super().rd(t[1:])) & M32
        return super().rd(t)

    def run(self, text):
        t = text.rstrip(" ;")
        op, rest = (t.split(None, 1) + [""])[:2]
        a = [x.strip() for x in rest.split(",")] if rest else []
        if op == "NOP":
            return
        if op.startswith("IADD3"):
            # qgen's emu, but "~R" is a plain bitwise not (no +1) and predicate inputs may be negated
            d = int(a[0][1:]) if a[0] != "RZ" else None
            k = 1; pouts = []
            while re.match(r"P[\dT]$", a[k]): pouts.append(a[k]); k += 1
            s = self.rd(a[k]) + self.rd(a[k + 1]) + self.rd(a[k + 2]) + self.negs(a[k:k + 3])
            if op.endswith(".X"): s += self.pred(a[k + 3]) + self.pred(a[k + 4])
            if d is not None: self.r[d] = s & M32
            c = s >> 32
            if len(pouts) == 1: self.setp(pouts[0], c)
            elif len(pouts) == 2: self.setp(pouts[0], min(c, 1)); self.setp(pouts[1], max(c - 1, 0))
        elif op.startswith("SHF.L.W.U32.HI"):  # d = (hi:lo << n) >> 32 with hi = a[3], lo = a[1]
            lo, n, hi = self.rd(a[1]), self.rd(a[2]) & 31, self.rd(a[3])
            self.r[int(a[0][1:])] = (((hi << 32 | lo) << n) >> 32) & M32
        elif op.startswith(("SHF.R.W.U32", "SHF.R.U32.HI", "SHF.R.U64")):  # d = (hi:lo) >> n (low word; .HI: high)
            lo, n, hi = self.rd(a[1]), self.rd(a[2]) & 31, self.rd(a[3])
            v = (hi << 32 | lo) >> n
            self.r[int(a[0][1:])] = (v >> 32 if op.startswith("SHF.R.U32.HI") else v) & M32
        elif op == "SEL":
            self.r[int(a[0][1:])] = self.rd(a[1]) if self.pred(a[3]) else self.rd(a[2])
        elif op in ("MOV", "IMAD.MOV.U32"):
            self.r[int(a[0][1:])] = self.rd(a[-1]) if op == "MOV" else (self.rd(a[1]) * self.rd(a[2]) + self.rd(a[3])) & M32
        elif op.startswith("LOP3.LUT"):
            x, y, z, lut = self.rd(a[1]), self.rd(a[2]), self.rd(a[3]), int(a[4], 16)
            v = 0
            for bit in range(32):
                i = (x >> bit & 1) << 2 | (y >> bit & 1) << 1 | (z >> bit & 1)
                v |= (lut >> i & 1) << bit
            self.r[int(a[0][1:])] = v
        elif op.startswith("ISETP"):  # ISETP.<cmp>[.U32].AND Pd, PT, a, b, PT
            cmp = op.split(".")[1]
            x, y = self.rd(a[2]), self.rd(a[3])
            if ".U32" not in op:
                x -= (x >> 31) << 32; y -= (y >> 31) << 32
            v = {"NE": x != y, "EQ": x == y, "LT": x < y, "LE": x <= y, "GT": x > y, "GE": x >= y}[cmp]
            self.setp(a[0], int(v) & self.pred(a[4]))
        elif op.startswith(("LDG", "STG")):  # [Rn.64+off] or desc[URx][Rn.64+off]
            load = op.startswith("LDG")
            m = re.search(r"\[R(\d+)\.64(?:\+(0x[0-9a-f]+))?\]", a[1] if load else a[0])
            base = self.r[int(m.group(1))] | (self.r[int(m.group(1)) + 1] << 32)
            addr = base + int(m.group(2) or "0", 16)
            n = 4 if ".128" in op else 2 if ".64" in op else 1
            if load:
                d = int(a[0][1:])
                for w in range(n): self.r[d + w] = self.mem.get(addr + 4 * w, 0)
            else:
                s = int(a[1][1:]) if a[1] != "RZ" else None
                for w in range(n): self.mem[addr + 4 * w] = self.r[s + w] if s is not None else 0
        else:
            super().run(text)
