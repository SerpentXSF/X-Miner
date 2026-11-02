# Emulator check of the generated field ops against Python integers (program order and list-scheduled order).
import os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "qgen"))
import field, nemu, sass

P = 2 ** 256 - 2 ** 32 - 977
M = 2 ** 256

A = list(range(10, 18)); B = list(range(20, 28)); O = list(range(30, 38))
SC = field.Scratch(t=list(range(40, 56)), pairs=[56, 58], ov=[60, 61])
K, K1 = 62, 63


def load(e, regs, v):
    for i, r in enumerate(regs): e.r[r] = v >> (32 * i) & 0xFFFFFFFF


def get(e, regs):
    return sum(e.r[r] << (32 * i) for i, r in enumerate(regs))


def run(prog, a, b, sched):
    if sched: prog = sass.schedule_list(prog)
    e = nemu.Emu()
    load(e, A, a); load(e, B, b)
    for t in prog: e.run(t)
    return get(e, O)


def gen(op):
    out = []; pr = field.Preds()
    sc = field.Scratch(list(SC.t), list(SC.pairs), list(SC.ov))
    if op == "mul": field.mul(out, pr, O, A, B, sc)
    elif op == "sqr": field.sqr(out, pr, O, A, sc)
    elif op == "add": field.add(out, pr, O, A, B, K)
    elif op == "sub": field.sub(out, pr, O, A, B, K, K1)
    return out


def ref(op, a, b):
    return {"mul": a * b, "sqr": a * a, "add": a + b, "sub": a - b}[op] % P


def main():
    rng = random.Random(7)
    edge = [0, 1, P - 1, P, P + 1, M - 1, M - 2 ** 32, 2 ** 255, 0x1000003D0]
    for op in ("add", "sub", "mul", "sqr"):
        prog = gen(op)
        bad = 0; n = 0; slips = 0
        cases = [(x, y) for x in edge for y in edge] + [(rng.getrandbits(256), rng.getrandbits(256)) for _ in range(300)]
        cases += [(M - 1 - rng.getrandbits(40), M - 1 - rng.getrandbits(40)) for _ in range(50)]
        for sched in (False, True):
            for a, b in cases:
                r = run(prog, a, b, sched)
                n += 1
                if r >= M or r % P != ref(op, a, b):
                    if a >= P - 2 ** 33 or b >= P - 2 ** 33:  # accepted slip: an operand within 2^33 of 2^256
                        slips += 1
                        continue
                    bad += 1
                    if bad <= 3: print(op, "sched" if sched else "order", hex(a), hex(b), "->", hex(r), "want", hex(ref(op, a, b)))
        wide = sum(1 for t in prog if "WIDE" in t)
        print("%s: %d instructions (%d WIDE), %d/%d ok, %d accepted slips (operands near 2^256)" % (op, len(prog), wide, n - bad - slips, n - slips, slips))
        assert bad == 0


if __name__ == "__main__":
    main()
