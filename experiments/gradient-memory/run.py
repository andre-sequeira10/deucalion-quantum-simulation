#!/usr/bin/env python3
"""Sequential fresh processes, bounded by per-process time and RSS guards."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def run_one(root, n, groups, method, rep):
    key = f'n{n:02d}-g{groups*5:03d}-{method}-r{rep}'
    result = root / (key+'.json')
    if result.exists():
        raise RuntimeError(f'Refusing to overwrite completed run {result}')
    cmd = [sys.executable, '-u', 'measure.py', '--qubits', str(n), '--groups', str(groups),
           '--method', method, '--repeat', str(rep), '--output', str(result)]
    start = time.monotonic()
    with (root/(key+'.stdout')).open('w') as out, (root/(key+'.stderr')).open('w') as err, (root/(key+'.rss.csv')).open('w') as samples:
        samples.write('elapsed_seconds,rss_bytes,hwm_bytes\n')
        p = subprocess.Popen(cmd, stdout=out, stderr=err)
        peak = 0
        while p.poll() is None:
            try:
                stat = dict(line.split(':',1) for line in Path(f'/proc/{p.pid}/status').read_text().splitlines())
                rss = int(stat['VmRSS'].split()[0])*1024
                hwm = int(stat['VmHWM'].split()[0])*1024
                peak = max(peak, rss)
                samples.write(f'{time.monotonic()-start:.6f},{rss},{hwm}\n')
                if rss > 24*2**30 or time.monotonic()-start > 900:
                    p.kill()
                    p.wait()
                    raise RuntimeError(f'Guard stopped {key}: RSS or 15-minute limit')
            except (FileNotFoundError, KeyError, ProcessLookupError):
                pass
            time.sleep(.01)
        rc = p.wait()
    record = dict(key=key, command=cmd, returncode=rc, total_seconds=time.monotonic()-start,
                  externally_sampled_lifetime_peak_bytes=peak)
    (root/(key+'.process.json')).write_text(json.dumps(record, indent=2)+'\n')
    if rc:
        raise RuntimeError(f'{key} exited {rc}: '+(root/(key+'.stderr')).read_text()[-3000:])
    print(json.dumps(record), flush=True)


def validate(root):
    import numpy as np
    rows = [json.loads(p.read_text()) for p in root.glob('n*.json') if not p.name.endswith('.process.json')]
    worst = 0.
    for row in rows:
        assert row['hwm_reset_error'] is None, row
        assert row['state_dtype'] == 'complex128' and row['affinity'] == [row['affinity_before'][0]]
        assert row['end']['VmHWM'] >= row['baseline']['VmRSS']
        assert row['baseline']['VmHWM'] <= row['baseline']['VmRSS'] + 2**20, row['baseline']
        peers = [r for r in rows if (r['qubits'],r['groups'],r['repeat']) == (row['qubits'],row['groups'],row['repeat'])]
        for other in peers:
            assert row['angles_sha256'] == other['angles_sha256']
            assert abs(row['value']-other['value']) < 1e-10
            if row['gradient'] is not None and other['gradient'] is not None:
                err = float(np.max(np.abs(np.array(row['gradient'])-other['gradient'])))
                worst = max(worst,err)
                assert err < 1e-9, (row['method'],other['method'],err)
    result = dict(completed=len(rows), max_gradient_absolute_difference=worst, checks_passed=True)
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['smoke','production'])
    args=ap.parse_args()
    root=Path(args.phase+'-results')
    root.mkdir(exist_ok=True)
    # The worker pins itself to one core; the low-cost sampler remains schedulable.
    affinity=sorted(os.sched_getaffinity(0))
    assert len(affinity)>=2
    methods=['backprop','adjoint','parameter-shift','forward']
    if args.phase=='smoke':
        cases=[(n,20,0) for n in (6,16)]
    else:
        smoke=json.loads(Path('smoke-results/validation.json').read_text())
        assert smoke['checks_passed'] and smoke['completed']==8
        cases=[(n,20,r) for r in range(3) for n in (6,8,10,12,14,16,18,20)]
        cases += [(16,g,r) for r in range(3) for g in (10,40)]
    schedule=[]
    for n,g,r in cases:
        order=methods.copy()
        random.Random(20261102+n+100*g+r).shuffle(order)
        schedule.extend((n,g,m,r) for m in order)
    (root/'schedule.json').write_text(json.dumps(schedule,indent=2)+'\n')
    for n,g,m,r in schedule:
        run_one(root,n,g,m,r)
    validate(root)


if __name__=='__main__':
    main()
