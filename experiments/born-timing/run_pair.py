"""Run all thread counts for one matched seed, sequentially on the same node."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--index', type=int, required=True)
p.add_argument('--pilot', action='store_true')
a = p.parse_args()
order = [1, 8, 16, 32, 48]
random.Random(20261027 + a.index//5).shuffle(order)
offset = a.index % 5
order = order[offset:] + order[:offset]
seed = 20261027 + a.index
root = Path('pilot' if a.pilot else 'results') / ('seed%d' % seed)
root.mkdir(parents=True, exist_ok=False)
(root/'order.json').write_text(json.dumps(dict(seed=seed, threads=order, index=a.index), indent=2)+'\n')
for threads in order:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_PROC_BIND='spread', OMP_PLACES='cores',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    command = [sys.executable, '-u', 'train_policy.py', '--threads', str(threads), '--seed', str(seed),
               '--output', str(root / ('t%d'%threads))]
    if a.pilot:
        command += ['--steps', '128', '--probe-steps', '8', '--eval-episodes', '2']
    start = time.perf_counter()
    with (root/('t%d.log'%threads)).open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (root/('t%d-process.json'%threads)).write_text(json.dumps(dict(process_wall_seconds=time.perf_counter()-start))+'\n')
(root/'complete.json').write_text(json.dumps(dict(status='complete', seed=seed, threads=order))+'\n')
