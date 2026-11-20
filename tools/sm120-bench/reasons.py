#!/usr/bin/env python3
"""reasons.py <label> [N]: per-instruction stall reasons from p_<label>.ncu-repz (not-issued warp samples)."""
import csv, re, subprocess, sys, collections
lab = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
txt = subprocess.run(['ncu', '--import', f'p_{lab}.ncu-repz', '--page', 'raw', '--print-metric-instances', 'details'], capture_output=True, text=True).stdout
R = collections.defaultdict(dict)
for m in re.finditer(r'\n\s+smsp__pcsamp_warps_issue_stalled_(\w+?)_not_issued\s+\S+\s+[\d\s ]+\(([^)]*)\)', txt):
    for a, v in re.findall(r'(0x[0-9a-f]+):\s*([\d\s ]+)', m.group(2)):
        R[m.group(1)][int(a, 16)] = int(re.sub(r'\D', '', v) or 0)
rows = list(csv.reader(open(f'p_{lab}.csv'))); hdr = rows[1]; data = [r for r in rows[2:] if len(r) > 6]
iS, iA = hdr.index("Source"), hdr.index("Address")
addr = [int(r[iA], 16) if r[iA].startswith('0x') else int(r[iA], 16) for r in data]
src = {a: (i, data[i][iS]) for i, a in enumerate(addr)}
tot = sum(sum(d.values()) for d in R.values())
by = {k: sum(d.values()) for k, d in R.items() if sum(d.values())}
print("not-issued stall samples by reason:", ", ".join(f"{k} {100*v/tot:.1f}%" for k, v in sorted(by.items(), key=lambda kv: -kv[1])))
bad = [k for k in by if k not in ('math_pipe_throttle',)]
score = collections.Counter()
for k in bad:
    for a, v in R[k].items(): score[a] += v
print(f"stalls other than waiting for the tensor unit: {100*sum(score.values())/tot:.1f}% of samples; top sites:")
for a, sc in score.most_common(N):
    i, s = src.get(a, (None, '?'))
    why = ", ".join(f"{k} {100*R[k].get(a,0)/tot:.2f}%" for k in sorted(bad, key=lambda k: -R[k].get(a, 0)) if R[k].get(a, 0) * 400 > tot)
    prev = " | ".join(x[iS][:34] for x in data[max(0, i - 3):i]) if i is not None else ''
    print(f"  {100*sc/tot:5.2f}%  [{why}]\n          after: {prev}\n          >> {s[:70]}")
