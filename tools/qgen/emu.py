# Minimal emulator for the SASS subset the Quantus generator emits (one thread), to check generated code against
# exact arithmetic before it is assembled. Registers R0..R255 (RZ = 0), predicates P0..P6 (PT = 1), c[] by dict.
import re
M32 = 0xFFFFFFFF
P = 0xFFFFFFFF00000001

class Emu:
    def __init__(self, const):
        self.r = [0] * 256; self.p = [0] * 7; self.c = const
    def rd(self, tok):
        tok = tok.replace(".reuse", "").strip()
        neg = tok.startswith("-"); tok = tok.lstrip("-")
        if tok == "RZ": v = 0
        elif tok.startswith("R"): v = self.r[int(tok[1:])]
        elif tok.startswith("c["): v = self.c[tok]
        else: return (-int(tok, 16) if neg else int(tok, 16)) & M32  # immediates are literal 32-bit values
        return (~v) & M32 if neg else v  # negation = ~v + 1, the +1 is added by the caller (carry like hardware)
    def negs(self, toks):
        return sum(1 for x in toks if x.strip().startswith("-R"))
    def rd64(self, tok):
        tok = tok.replace(".reuse", "").strip()
        if tok == "RZ": return 0
        n = int(tok[1:]); return self.r[n] | (self.r[n + 1] << 32)
    def pred(self, tok):
        tok = tok.strip(); neg = tok.startswith("!"); tok = tok.lstrip("!")
        v = 1 if tok == "PT" else self.p[int(tok[1:])]
        return v ^ 1 if neg else v
    def setp(self, tok, v):
        if tok != "PT": self.p[int(tok[1:])] = v & 1
    def run(self, text):
        t = text.rstrip(" ;")
        op, rest = t.split(None, 1); a = [x.strip() for x in rest.split(",")]
        if op.startswith("IMAD.WIDE.U32"):
            d = int(a[0][1:]); k = 1; pout = None
            if re.match(r"P\d$", a[1]): pout = a[1]; k = 2
            x, y, z = self.rd(a[k]), self.rd(a[k + 1]), self.rd64(a[k + 2])
            s = x * y + z + (self.pred(a[k + 3]) if op.endswith(".X") else 0)
            self.r[d], self.r[d + 1] = s & M32, (s >> 32) & M32
            if pout: self.setp(pout, s >> 64)
        elif op.startswith("IADD3"):
            d = int(a[0][1:]) if a[0] != "RZ" else None; k = 1; pouts = []
            while re.match(r"P[\dT]$", a[k]): pouts.append(a[k]); k += 1
            s = self.rd(a[k]) + self.rd(a[k + 1]) + self.rd(a[k + 2]) + self.negs(a[k:k + 3])
            if op.endswith(".X"): s += self.pred(a[k + 3]) + self.pred(a[k + 4])
            if d is not None: self.r[d] = s & M32
            # two carry outputs: their sum is the carry count (each 0/1)
            c = s >> 32
            if len(pouts) == 1: self.setp(pouts[0], c)
            elif len(pouts) == 2: self.setp(pouts[0], min(c, 1)); self.setp(pouts[1], max(c - 1, 0))
        elif op.startswith("IMAD.X"):
            d = int(a[0][1:]); s = self.rd(a[1]) * self.rd(a[2]) + self.rd(a[3]) + self.pred(a[4])
            self.r[d] = s & M32
        elif op.startswith("IMAD"):
            d = int(a[0][1:]); self.r[d] = (self.rd(a[1]) * self.rd(a[2]) + self.rd(a[3])) & M32
        elif op == "MOV":
            self.r[int(a[0][1:])] = self.rd(a[1])
        elif op.startswith("ISETP.NE.AND"):  # ISETP.NE.AND Pd, PT, Ra, b, PT
            self.setp(a[0], int(self.rd(a[2]) != self.rd(a[3])))
        elif op == "NOP":
            pass
        elif op.startswith("LDC"):  # LDC.64 Rd, c[bank][Rx+0xoff]
            m = re.match(r"c\[(0x[0-9a-f]+)\]\[R(\d+)(?:\+(0x[0-9a-f]+))?\]", a[1])
            addr = self.r[int(m.group(2))] + int(m.group(3) or "0", 16)
            d = int(a[0][1:])
            for w in range(2 if op.endswith(".64") else 1):
                self.r[d + w] = self.c.get("c[%s][0x%x]" % (m.group(1), addr + 4 * w), 0)
        else:
            raise ValueError("unsupported: " + text)

    def run_program(self, lines, limit=10 ** 7):
        """Runs ctrl-annotated lines (labels, @Pn guards, BRA) from the first line until it falls off the end."""
        prog, labels = [], {}
        for l in lines:
            t = re.sub(r"^\s*\[[^\]]*\]\s*", "", l).strip()
            if t.endswith(":"): labels[t[:-1]] = len(prog)
            elif t: prog.append(t)
        pc, steps = 0, 0
        while pc < len(prog):
            t = prog[pc]; pc += 1; steps += 1
            assert steps < limit, "runaway loop"
            g = re.match(r"^@(!?P\d)\s+", t)
            if g:
                t = t[g.end():]
                if not self.pred(g.group(1)): continue
            if t.startswith("BRA"):
                pc = labels[re.search(r"`\((\S+)\)", t).group(1)]; continue
            self.run(t)
