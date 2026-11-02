# secp256k1 field arithmetic (p = 2^256 - 2^32 - 977) as sm_86 SASS for the Nexa generator. A field element is 8
# registers holding 32-bit little-endian words; values are any 256-bit number congruent mod p (results < 2^256, not
# canonical). Rare slips (a dropped final carry, probability below 2^-200) are accepted: they only lose a nonce,
# and the CPU re-checks every candidate.
#
# Each op appends instruction strings (no control codes) to a list; sass.py schedules them. Ops take their scratch
# registers explicitly. Carry predicates come from the Preds rotation (P0..P5; P6 stays free for loops).

C3D1 = 0x3D1  # 2^256 = 2^32 + 0x3D1 (mod p)


class Preds:
    def __init__(self, names=("P0", "P1", "P2", "P3", "P4", "P5")):
        self.names, self.i = list(names), 0

    def get(self):
        p = self.names[self.i % len(self.names)]
        self.i += 1
        return p


class Scratch:
    """Registers a multiply / square may use: t (16 product words), pairs (even registers r, r + 1 free) for the
    64-bit accumulators, ov (2 overflow counters)."""
    def __init__(self, t, pairs, ov):
        assert len(t) == 16 and len(pairs) >= 2 and len(ov) == 2
        assert all(r % 2 == 0 for r in pairs)
        self.t, self.pairs, self.ov = t, pairs, ov


def _count(out, preds, carries, ov, first):
    """ov (+)= the carry predicates (two per IADD3.X). Returns whether ov now holds a value."""
    while carries:
        c0 = carries.pop(0)
        c1 = carries.pop(0) if carries else "!PT"
        src = "RZ" if first else "R%d" % ov
        out.append("IADD3.X R%d, %s, RZ, RZ, %s, %s" % (ov, src, c0, c1))
        first = False
    return not first


def _odd_first(a, b):
    """Operand order for IMAD.WIDE (GA106 register banks = register parity): an odd register in the first slot (an
    even one there costs an issue cycle when the addend is a 64-bit pair)."""
    return (b, a) if a % 2 == 0 and b % 2 == 1 else (a, b)


def _columns(out, preds, sc, prods_by_col, ncols, mov=True):
    """Product scanning: column k = sum of its products + the previous column's (hi, ov). The previous column's part is
    added last, so that a column's products do not wait for the column before (two columns' multiply chains overlap;
    their accumulators alternate). Writes word k to sc.t[k] (by MOV, or leaves it in the accumulator: then sc.t[k] is
    rebound). Returns the registers of the last hi word and its overflow count (None: zero)."""
    prev_hi, prev_ov = None, None
    for k in range(ncols):
        prods = prods_by_col[k]
        E = sc.pairs[k % len(sc.pairs)]
        O = sc.ov[k % 2]
        carries = []
        has_ov = False
        if prods:
            (a0, b0), rest = _odd_first(*prods[0]), prods[1:]
            out.append("IMAD.WIDE.U32 R%d, R%d, R%d, RZ" % (E, a0, b0))
            for (a, b) in rest:
                a, b = _odd_first(a, b)
                p = preds.get()
                out.append("IMAD.WIDE.U32 R%d, %s, R%d, R%d, R%d" % (E, p, a, b, E))
                carries.append(p)
                if len(carries) == 2:
                    has_ov = _count(out, preds, carries, O, not has_ov)
        else:  # no products: the word is the previous column's (hi, ov)
            out.append("MOV R%d, RZ" % E)
            out.append("MOV R%d, RZ" % (E + 1))
        if prev_hi is not None:
            pa, pb = preds.get(), preds.get()
            out.append("IADD3 R%d, %s, R%d, R%d, RZ" % (E, pa, E, prev_hi))
            out.append("IADD3.X R%d, %s, R%d, %s, RZ, %s, !PT" % (E + 1, pb, E + 1,
                                                                  "R%d" % prev_ov if prev_ov is not None else "RZ", pa))
            carries.append(pb)
        has_ov = _count(out, preds, carries, O, not has_ov)
        if mov:
            out.append("MOV R%d, R%d" % (sc.t[k], E))
        else:
            sc.t[k] = E
        prev_hi, prev_ov = E + 1, (O if has_ov else None)
    return prev_hi, prev_ov


def reduce512(out, preds, o, t, pairs):
    """o = t (16 words, 512 bits) mod p, < 2^256: o_i = t_i + 0x3D1 t_(8+i) + t_(7+i) + carry, then the top word
    folded back once more."""
    prev = None
    for i in range(8):
        E = pairs[i % len(pairs)]
        pa, pb = preds.get(), preds.get()
        out.append("IADD3 R%d, %s, %s, R%d, %s, %s" % (E, pa, pb, t[i], "R%d" % t[7 + i] if i else "RZ",
                                                        "R%d" % prev if prev is not None else "RZ"))
        out.append("IADD3.X R%d, RZ, RZ, RZ, %s, %s" % (E + 1, pa, pb))
        out.append("IMAD.WIDE.U32 R%d, R%d, 0x%x, R%d" % (E, t[8 + i], C3D1, E))
        out.append("MOV R%d, R%d" % (o[i], E))
        prev = E + 1
    # word 8 = t15 + carry (< 2^33): v + vh 2^32, folded as (v + vh 2^32)(2^32 + 0x3D1)
    E = pairs[8 % len(pairs)]
    F = pairs[9 % len(pairs)]
    pv = preds.get()
    out.append("IADD3 R%d, %s, R%d, R%d, RZ" % (E, pv, t[15], prev))
    out.append("IADD3.X R%d, RZ, RZ, RZ, %s, !PT" % (E + 1, pv))              # vh
    out.append("IMAD.WIDE.U32 R%d, R%d, 0x%x, RZ" % (F, E, C3D1))             # v 0x3D1
    out.append("IMAD R%d, R%d, 0x%x, R%d" % (F + 1, E + 1, C3D1, F + 1))      # + vh 0x3D1 2^32 (hi word)
    p1 = preds.get()
    out.append("IADD3 R%d, %s, R%d, R%d, RZ" % (o[0], p1, o[0], F))
    p2, p3 = preds.get(), preds.get()
    out.append("IADD3.X R%d, %s, R%d, R%d, RZ, %s, !PT" % (o[1], p2, o[1], F + 1, p1))
    out.append("IADD3 R%d, %s, R%d, R%d, RZ" % (o[1], p3, o[1], E))                    # + v (2^32 term)
    p4 = preds.get()
    out.append("IADD3.X R%d, %s, R%d, R%d, RZ, %s, %s" % (o[2], p4, o[2], E + 1, p2, p3))  # + vh (2^64 term)
    prevp = p4
    for j in range(3, 8):
        p = preds.get()
        out.append("IADD3.X R%d, %s, R%d, RZ, RZ, %s, !PT" % (o[j], p, o[j], prevp))
        prevp = p


def mul(out, preds, o, a, b, sc):
    cols = [[(a[i], b[k - i]) for i in range(8) if 0 <= k - i < 8] for k in range(15)]
    hi, ov = _columns(out, preds, sc, cols, 15)
    out.append("MOV R%d, R%d" % (sc.t[15], hi))  # word 15 (the product is < 2^512, so the last ov is 0)
    reduce512(out, preds, o, sc.t, sc.pairs)


def sqr(out, preds, o, a, sc):
    """Cross products a_i a_j (i < j) by columns into words 1..15, doubled in place (top down, no carries), then
    a_i^2 added two words at a time with a carry chain. sc.t must be 8 register pairs (t[2i], t[2i + 1] = r, r + 1)."""
    t = sc.t
    assert all(t[2 * i] % 2 == 0 and t[2 * i + 1] == t[2 * i] + 1 for i in range(8))
    cols = [[(a[i], a[k - i]) for i in range(8) if i < k - i < 8] for k in range(1, 14)]
    sc.t = t[1:14] + [t[14], t[15], t[0]]
    hi, ov = _columns(out, preds, sc, cols, 13)  # words 1..13
    sc.t = t
    out.append("MOV R%d, R%d" % (t[14], hi))
    out.append("MOV R%d, %s" % (t[15], "R%d" % ov if ov is not None else "RZ"))
    for j in range(15, 0, -1):
        out.append("SHF.L.W.U32.HI R%d, %s, 0x1, R%d" % (t[j], "R%d" % t[j - 1] if j > 1 else "RZ", t[j]))
    out.append("MOV R%d, RZ" % t[0])  # word 0 of the cross sum
    cin = None
    for i in range(8):
        lo = t[2 * i]
        pw = preds.get()
        out.append("IMAD.WIDE.U32 R%d, %s, R%d, R%d, R%d" % (lo, pw, a[i], a[i], lo))
        c = sc.ov[i % 2]
        if cin is not None:
            px, py = preds.get(), preds.get()
            out.append("IADD3 R%d, %s, R%d, R%d, RZ" % (lo, px, lo, cin))
            out.append("IADD3.X R%d, %s, R%d, RZ, RZ, %s, !PT" % (lo + 1, py, lo + 1, px))
            if i < 7: out.append("IADD3.X R%d, RZ, RZ, RZ, %s, %s" % (c, pw, py))
        elif i < 7:
            out.append("IADD3.X R%d, RZ, RZ, RZ, %s, !PT" % (c, pw))
        cin = c
    reduce512(out, preds, o, t, sc.pairs)


def add(out, preds, o, a, b, k):
    """o = a + b mod p (< 2^256). k: one scratch register."""
    p = preds.get()
    out.append("IADD3 R%d, %s, R%d, R%d, RZ" % (o[0], p, a[0], b[0]))
    for j in range(1, 8):
        q = preds.get()
        out.append("IADD3.X R%d, %s, R%d, R%d, RZ, %s, !PT" % (o[j], q, a[j], b[j], p))
        p = q
    out.append("SEL R%d, RZ, 0x%x, !%s" % (k, C3D1, p))  # carry out of 2^256: + 0x1000003D1
    q = preds.get()
    out.append("IADD3 R%d, %s, R%d, R%d, RZ" % (o[0], q, o[0], k))
    r = preds.get()
    out.append("IADD3.X R%d, %s, R%d, RZ, RZ, %s, %s" % (o[1], r, o[1], q, p))
    for j in range(2, 8):
        s = preds.get()
        out.append("IADD3.X R%d, %s, R%d, RZ, RZ, %s, !PT" % (o[j], s, o[j], r))
        r = s


def sub(out, preds, o, a, b, k, k1):
    """o = a - b mod p (< 2^256). k, k1: scratch registers."""
    p = preds.get()
    out.append("IADD3 R%d, %s, R%d, -R%d, RZ" % (o[0], p, a[0], b[0]))
    for j in range(1, 8):
        q = preds.get()
        out.append("IADD3.X R%d, %s, R%d, ~R%d, RZ, %s, !PT" % (o[j], q, a[j], b[j], p))
        p = q
    # p = 0: borrow (a < b): the result is a - b + 2^256, so subtract 0x1000003D1
    out.append("SEL R%d, RZ, 0x%x, %s" % (k, C3D1, p))
    out.append("SEL R%d, RZ, 0x1, %s" % (k1, p))
    q = preds.get()
    out.append("IADD3 R%d, %s, R%d, -R%d, RZ" % (o[0], q, o[0], k))
    r = preds.get()
    out.append("IADD3.X R%d, %s, R%d, ~R%d, RZ, %s, !PT" % (o[1], r, o[1], k1, q))
    for j in range(2, 8):
        s = preds.get()
        out.append("IADD3.X R%d, %s, R%d, 0xffffffff, RZ, %s, !PT" % (o[j], s, o[j], r))
        r = s
