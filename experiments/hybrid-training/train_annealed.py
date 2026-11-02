"""Standalone PennyLane learning pilot: trainable embedding, entangled observables, softmax.

This is a fresh implementation of the documented historical architecture family.
It does not import any project-specific model or training package.
"""
import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

LAUNCH_AFFINITY=sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None

import gymnasium as gym
import numpy as np
import pennylane as qml
import torch
from torch import nn


def save(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def qnode_for(device, method):
    @qml.qnode(device, interface='torch', diff_method=method)
    def observables(rotations, embedding, state):
        for w in range(6):
            qml.Hadamard(wires=w)
        for layer in range(3):
            for w in range(6):
                qml.RZ(rotations[layer,0,w],wires=w)
                qml.RY(rotations[layer,1,w],wires=w)
            if layer<2:
                for a in range(6):
                    for b in range(a+1,6):
                        qml.CZ(wires=[a,b])
            # One learned RZ embedding between the first and second layers.
            if layer==0:
                angles=embedding @ state
                for w in range(6):
                    qml.RZ(angles[w],wires=w)
        return ([qml.expval(qml.PauliZ(w)) for w in range(6)]
                +[qml.expval(qml.PauliZ(w)@qml.PauliZ(w+1)) for w in range(5)]
                +[qml.expval(qml.prod(*(qml.PauliZ(w) for w in range(6))))])
    return observables


class Policy(nn.Module):
    def __init__(self, device, method='adjoint'):
        super().__init__()
        self.rotations=nn.Parameter(torch.randn(3,2,6,dtype=torch.float64))
        self.embedding=nn.Parameter(torch.randn(6,4,dtype=torch.float64))
        self.readout=nn.Linear(12,2,dtype=torch.float64)
        self.circuit=qnode_for(device,method)
        self.register_buffer('scale',torch.tensor([2.4,3.,.2095,3.5],dtype=torch.float64))

    def forward(self, observation):
        state=torch.clamp(torch.as_tensor(observation,dtype=torch.float64)/self.scale,-3.,3.)
        values=torch.stack(self.circuit(self.rotations,self.embedding,state))
        return self.readout(values)


def evaluate(policy, seed, episodes, limit):
    env=gym.make('CartPole-v1',max_episode_steps=limit)
    rng=np.random.default_rng(seed+9000000+limit)
    returns=[]
    with torch.no_grad():
        for episode in range(episodes):
            obs,_=env.reset(seed=seed+episode)
            total=0
            for _ in range(limit):
                probs=torch.softmax(policy(obs),dim=-1).numpy()
                assert np.isfinite(probs).all() and abs(probs.sum()-1)<1e-10
                action=int(rng.random()>=probs[0])
                obs,reward,done,trunc,_=env.step(action)
                total+=float(reward)
                if done or trunc: break
            returns.append(total)
    env.close()
    return returns


def workers():
    result={}
    for d in Path('/proc/self/task').iterdir():
        try:
            stat=(d/'stat').read_text().split(') ')[1].split()
            result[d.name]=dict(ticks=int(stat[11])+int(stat[12]),cpu=int(stat[36]),affinity=sorted(os.sched_getaffinity(int(d.name))))
        except (FileNotFoundError,ProcessLookupError):pass
    return result


def rate_factor(steps, start, end, final_factor):
    return final_factor ** min(1., max(0., (steps-start)/(end-start)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--transitions',type=int,default=80000);ap.add_argument('--limit',type=int,default=200)
    ap.add_argument('--batch-episodes',type=int,default=5);ap.add_argument('--eval-episodes',type=int,default=100)
    ap.add_argument('--eval-every',type=int,default=10000);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--decay-start',type=int,default=30000);ap.add_argument('--decay-end',type=int,default=80000);ap.add_argument('--final-rate-factor',type=float,default=.1)
    a=ap.parse_args();assert 0<=a.decay_start<a.decay_end<=a.transitions and 0<a.final_rate_factor<=1
    assert min(a.transitions,a.limit,a.batch_episodes,a.threads,a.eval_episodes)>0
    assert int(os.environ['OMP_NUM_THREADS'])==a.threads
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_default_dtype(torch.float64);torch.set_num_interop_threads(1);torch.manual_seed(a.seed)
    # PyTorch import can reset shared libgomp's default to1. Restore before Kokkos init.
    torch.set_num_threads(a.threads)
    libpaths=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libgomp' in line or 'libomp.' in line})
    for path in libpaths:ctypes.CDLL(path).omp_set_num_threads(a.threads)
    if LAUNCH_AFFINITY is not None:os.sched_setaffinity(0,LAUNCH_AFFINITY)
    from pennylane_lightning.lightning_kokkos_ops import InitializationSettings
    settings=InitializationSettings();settings.set_num_threads(a.threads)
    dev=qml.device('lightning.kokkos',wires=6,shots=None,c_dtype=np.complex128,mpi=False,kokkos_args=settings)
    policy=Policy(dev);check=np.array([.1,.2,.03,-.4])
    with torch.no_grad():policy(check)
    torch.set_num_threads(1)
    # Keep the main worker on the first OpenMP place; other workers are spread-bound.
    if LAUNCH_AFFINITY is not None:os.sched_setaffinity(0,{LAUNCH_AFFINITY[0]})
    reference=Policy(qml.device('default.qubit',wires=6,shots=None),'backprop');reference.load_state_dict(policy.state_dict())
    def gradient(model):
        model.zero_grad(set_to_none=True);logits=model(check)
        (-1.3*torch.log_softmax(logits,dim=-1)[1]).backward()
        return logits.detach().numpy(),{n:p.grad.detach().clone() for n,p in model.named_parameters()}
    logits,g=gradient(policy);rl,rg=gradient(reference)
    errs={n:float(torch.max(abs(g[n]-rg[n]))) for n in g}
    assert np.allclose(logits,rl,atol=1e-9,rtol=1e-8) and max(errs.values())<1e-8
    assert all(torch.isfinite(v).all() and torch.linalg.vector_norm(v)>1e-9 for v in g.values())
    # Fixed-input probes and worker placement are outside training timing.
    before=workers();tick=time.perf_counter()
    with torch.no_grad():
        for _ in range(100):policy(check)
    forward_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    for _ in range(20):gradient(policy)
    adjoint_seconds=time.perf_counter()-tick;after=workers()
    config=dict(qubits=6,layers=3,parameters=86,transitions=a.transitions,episode_limit=a.limit,batch_episodes=a.batch_episodes,gamma=.99,optimizer='Adam AMSGrad',quantum_embedding_lr=.1,readout_lr=.01,freeze=False,rate_schedule=dict(kind='exponential_by_batch_start_transitions',start=a.decay_start,end=a.decay_end,final_factor=a.final_rate_factor),final_partial_batch='Update on collected episode segments; final segment return truncated at fixed transition budget',model='H, RZ-RY layers, all-pairs CZ between layers, learned RZ embedding after first CZ, 12 Pauli observables, linear softmax')
    import re
    kokkos=dev._statevector._kokkos_configuration()
    pool=re.search(r'thread_pool_topology\[\s*1\s*x\s*(\d+)\s*x\s*1\s*\]',kokkos)
    assert pool and int(pool.group(1))==a.threads and torch.get_num_threads()==1
    meta=dict(seed=a.seed,threads=a.threads,config=config,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions={n:importlib.metadata.version(n) for n in ['pennylane','pennylane-lightning-kokkos','numpy','gymnasium','torch']},kokkos_configuration=kokkos,torch_threads=torch.get_num_threads(),launch_cpu_affinity=LAUNCH_AFFINITY,main_cpu_affinity=sorted(os.sched_getaffinity(0)),job_id=os.getenv('SLURM_JOB_ID'),host=platform.node(),gradient_errors=errs,initial_gradients={n:v.tolist() for n,v in g.items()},workers_before=before,workers_after=after,forward_probe_seconds=forward_seconds,forward_probe_calls=100,adjoint_probe_seconds=adjoint_seconds,adjoint_probe_calls=20,environment={n:os.getenv(n) for n in ['OMP_NUM_THREADS','OMP_PLACES','OMP_PROC_BIND','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    save(a.output/'metadata.json',meta)
    initial={n:p.detach().tolist() for n,p in policy.named_parameters()}
    save(a.output/'initial-parameters.json',initial)
    optimizer=torch.optim.Adam([dict(params=[policy.rotations,policy.embedding],lr=.1),dict(params=policy.readout.parameters(),lr=.01)],amsgrad=True)
    evals=[dict(steps=0,limit=500,returns=evaluate(policy,a.seed+1000000,10,500))]
    save(a.output/'evaluation.json',evals)
    env=gym.make('CartPole-v1',max_episode_steps=a.limit);rng=np.random.default_rng(a.seed+100000)
    steps=0;episodes=0;updates=0;baseline=0.;baseline_ready=False;train_seconds=0.;rollout_seconds=0.;gradient_seconds=0.;next_eval=a.eval_every
    action_hash=hashlib.sha256();obs_hash=hashlib.sha256();returns=[]
    ef=(a.output/'episodes.csv').open('w',newline='');ew=csv.writer(ef)
    ew.writerow(['episode','return','steps_total','training_seconds','updates','budget_truncated'])
    uf=(a.output/'updates.csv').open('w',newline='');uw=csv.writer(uf)
    uw.writerow(['update','steps_total','episodes','loss','gradient_norm','training_seconds','rollout_seconds','gradient_seconds','batch_start_steps','quantum_embedding_lr','readout_lr'])
    while steps<a.transitions:
        tick=time.perf_counter();batch_start_steps=steps
        factor=rate_factor(steps,a.decay_start,a.decay_end,a.final_rate_factor)
        optimizer.param_groups[0]['lr']=.1*factor;optimizer.param_groups[1]['lr']=.01*factor
        batch_logp=[];batch_adv=[]
        for _ in range(a.batch_episodes):
            if steps==a.transitions:break
            obs,_=env.reset(seed=a.seed+episodes);rewards=[];logps=[];done=False;trunc=False
            for _ in range(min(a.limit,a.transitions-steps)):
                obs_hash.update(np.asarray(obs,dtype='<f8').tobytes());logits=policy(obs)
                probs=torch.softmax(logits.detach(),dim=-1).numpy();assert np.isfinite(probs).all() and abs(probs.sum()-1)<1e-10
                action=int(rng.random()>=probs[0]);action_hash.update(bytes([action]))
                logps.append(torch.log_softmax(logits,dim=-1)[action])
                obs,reward,done,trunc,_=env.step(action);steps+=1;rewards.append(float(reward))
                if done or trunc:break
            episodes+=1;returns.append(sum(rewards));disc=[];future=0.
            for reward in rewards[::-1]:future=reward+.99*future;disc.append(future)
            disc=torch.tensor(disc[::-1]);batch_adv.append(disc-baseline);batch_logp.extend(logps)
            mean=float(disc.mean());baseline=mean if not baseline_ready else .95*baseline+.05*mean;baseline_ready=True
            ew.writerow([episodes,returns[-1],steps,train_seconds+time.perf_counter()-tick,updates,bool(steps==a.transitions and not(done or trunc))]);ef.flush()
        split=time.perf_counter();rollout_seconds+=split-tick
        advantage=torch.cat(batch_adv)
        std=advantage.std() if advantage.numel()>1 else torch.zeros(())
        advantage=(advantage-advantage.mean())/(std+1e-8)
        loss=-(advantage*torch.stack(batch_logp)).mean();optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=math.sqrt(sum(float(torch.sum(p.grad**2)) for p in policy.parameters()))
        assert math.isfinite(norm) and all(torch.isfinite(p.grad).all() for p in policy.parameters())
        optimizer.step();updates+=1;end=time.perf_counter();gradient_seconds+=end-split;train_seconds+=end-tick
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        uw.writerow([updates,steps,episodes,float(loss.detach()),norm,train_seconds,rollout_seconds,gradient_seconds,batch_start_steps,optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr']]);uf.flush()
        save(a.output/'progress.json',dict(steps=steps,target=a.transitions,episodes=episodes,updates=updates,training_seconds=train_seconds,mean_last100=float(np.mean(returns[-100:]))))
        if steps>=next_eval or steps==a.transitions:
            evals.append(dict(steps=steps,limit=500,returns=evaluate(policy,a.seed+2000000,10,500)));save(a.output/'evaluation.json',evals)
            torch.save(policy.state_dict(),a.output/f'checkpoint-{steps:06d}.pt');next_eval=(steps//a.eval_every+1)*a.eval_every
    ef.close();uf.close();env.close()
    save(a.output/'parameters.json',dict(initial=initial,final={n:p.detach().tolist() for n,p in policy.named_parameters()}))
    final={}
    for limit,offset in [(200,3000000),(500,4000000)]:
        rs=evaluate(policy,a.seed+offset,a.eval_episodes,limit)
        save(a.output/f'heldout_{limit}.json',dict(returns=rs,mean=float(np.mean(rs)),limit=limit))
        final[f'heldout_{limit}']=rs;final[f'heldout_mean_{limit}']=float(np.mean(rs))
    save(a.output/'summary.json',dict(status='complete',seed=a.seed,threads=a.threads,transitions=steps,episodes=episodes,updates=updates,training_seconds=train_seconds,rollout_seconds=rollout_seconds,gradient_seconds=gradient_seconds,action_sha256=action_hash.hexdigest(),observation_sha256=obs_hash.hexdigest(),passes_200=bool(final['heldout_mean_200']>=195),passes_500=bool(final['heldout_mean_500']>=475),**final))

if __name__=='__main__':main()
