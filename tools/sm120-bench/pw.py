#!/usr/bin/env python3
"""pw.py <seconds> <warmup> <cmd...>: run cmd in a loop for <seconds>, report mean board power after <warmup> s."""
import subprocess, sys, time
T, W = float(sys.argv[1]), float(sys.argv[2]); cmd = sys.argv[3:]
def q(): return [float(x) for x in subprocess.run(['nvidia-smi', '--query-gpu=power.draw,temperature.gpu,clocks.sm', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split(',')]
t0 = time.time(); ps = []; last = ''
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
while time.time() - t0 < T:
    if p.poll() is not None:
        last = (p.stdout.read().strip().splitlines() or [last])[-1]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    time.sleep(1.0)
    if time.time() - t0 > W: ps.append(q())
p.kill()
n = len(ps)
print(f"{sum(x[0] for x in ps)/n:6.1f} W  {sum(x[1] for x in ps)/n:4.1f} C  {sum(x[2] for x in ps)/n:.0f} MHz | {last[:100]}")
