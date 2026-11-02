"""Controlled CartPole training: one entangled parity policy, varying OpenMP threads."""
import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_score(device):
    @qml.qnode(device, interface='autograd', diff_method='adjoint')
    def score(weights, angles):
        # Six qubits, three data-reuploading layers, 36 trainable angles.
        for layer in range(3):
            for wire in range(6):
                qml.RY(angles[wire % 4], wires=wire)
                qml.RZ(weights[layer, wire, 0], wires=wire)
                qml.RY(weights[layer, wire, 1], wires=wire)
            # Alternating nearest-neighbour CNOT matchings, including wraparound.
            for control, target in [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (5, 0)]:
                qml.CNOT(wires=[control, target])
        return qml.expval(qml.prod(*(qml.PauliZ(w) for w in range(6))))
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--probe-steps', type=int, default=64)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', action='store_true')
    args = parser.parse_args()
    if args.steps % args.batch_size or min(args.steps, args.batch_size, args.threads,
                                          args.probe_steps, args.eval_episodes) < 1:
        raise ValueError('Positive sizes required; steps must be divisible by batch size')
    if int(os.environ.get('OMP_NUM_THREADS', '0')) != args.threads:
        raise ValueError('OMP_NUM_THREADS differs from requested thread count')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    setup = {}
    if args.reference:
        dev = qml.device('default.qubit', wires=6, shots=None)
    else:
        from pennylane_lightning.lightning_kokkos_ops import InitializationSettings
        kokkos = InitializationSettings()
        kokkos.set_num_threads(args.threads)
        dev = qml.device('lightning.kokkos', wires=6, shots=None,
                         c_dtype=np.complex128, mpi=False, kokkos_args=kokkos)
        omp = ctypes.CDLL('libgomp.so.1')
        omp.omp_get_max_threads.restype = ctypes.c_int
        setup['openmp_max_threads'] = omp.omp_get_max_threads()
        if setup['openmp_max_threads'] != args.threads:
            raise RuntimeError('OpenMP thread configuration mismatch')
    score = make_score(dev)

    def loss(weights, angles, action, advantage):
        probability = (1 + (1 - 2 * action) * score(weights, angles)) / 2
        return -advantage * pnp.log(pnp.clip(probability, 1e-12, 1.0))

    gradient = qml.grad(loss, 0)
    theta0 = np.random.default_rng(args.seed).uniform(-np.pi, np.pi, (3, 6, 2))
    theta = pnp.array(theta0, requires_grad=True)
    probe_rng = np.random.default_rng(8675309)
    probe_angles = np.arctan(probe_rng.normal(size=(args.probe_steps, 4)))
    probe_angles = [pnp.array(x, requires_grad=False) for x in probe_angles]
    # Warm up both forward and adjoint execution before either timed phase.
    for _ in range(3):
        score(theta, probe_angles[0])
        gradient(theta, probe_angles[0], 0, 1.0)
    z = float(score(theta, probe_angles[0]))
    g = np.asarray(gradient(theta, probe_angles[0], 0, 1.0))
    reference_score = make_score(qml.device('default.qubit', wires=6, shots=None))

    def reference_loss(weights):
        return -pnp.log((1 + reference_score(weights, probe_angles[0])) / 2)

    rz = float(reference_score(theta, probe_angles[0]))
    rg = np.asarray(qml.grad(reference_loss)(theta))
    if not np.allclose(g, rg, atol=1e-8, rtol=1e-7) or abs(z-rz) > 1e-10:
        raise RuntimeError('Policy/adjoint validation against independent simulator failed')
    if not np.isfinite(g).all() or np.linalg.norm(g) < 1e-10:
        raise RuntimeError('Invalid or zero initial policy gradient')
    if not args.reference:
        setup['kokkos_configuration'] = dev._statevector._kokkos_configuration()
    config = dict(qubits=6, layers=3, parameters=36, actions=2,
                  policy='global parity Born policy', environment='CartPole-v1',
                  max_episode_steps=500, training_steps=args.steps,
                  batch_size=args.batch_size, gamma=0.99, learning_rate=0.02,
                  optimizer='SGD, gradient norm clipped at 5',
                  return_estimator='discounted reward-to-go, truncated at batch/episode boundary; batch standardized',
                  precision='complex128', shots=None, diff_method='adjoint',
                  device=dev.name, mpi=False, seed=args.seed,
                  probe_steps=args.probe_steps, eval_episodes=args.eval_episodes)
    metadata = dict(config=config, threads=args.threads, host=platform.node(),
                    machine=platform.machine(), job_id=os.getenv('SLURM_JOB_ID'),
                    array_task=os.getenv('SLURM_ARRAY_TASK_ID'),
                    cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                    environment={k: os.getenv(k) for k in ['OMP_NUM_THREADS', 'OMP_PLACES', 'OMP_PROC_BIND',
                                 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']},
                    versions={k: importlib.metadata.version(k) for k in ['pennylane', 'numpy', 'gymnasium']},
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    validation=dict(score_error=abs(z-rz), gradient_max_error=float(np.max(np.abs(g-rg))),
                                    initial_score=z, initial_gradient=g.tolist()), **setup)
    save_json(args.output/'metadata.json', metadata)
    probe_start = time.perf_counter()
    for i, angles in enumerate(probe_angles):
        score(theta, angles)
        gradient(theta, angles, i % 2, 1.0)
    probe_seconds = time.perf_counter() - probe_start

    def evaluate(weights):
        env = gym.make('CartPole-v1', max_episode_steps=500)
        returns = []
        for episode in range(args.eval_episodes):
            eval_seed = args.seed + 1000000 + episode
            rng = np.random.default_rng(eval_seed)
            obs, _ = env.reset(seed=eval_seed)
            total = 0
            for _ in range(500):
                angles = pnp.array(np.arctan(obs), requires_grad=False)
                p0 = np.clip((1 + float(score(weights, angles))) / 2, 0, 1)
                action = int(rng.random() >= p0)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            returns.append(total)
        env.close()
        return returns

    initial_returns = evaluate(theta)
    env = gym.make('CartPole-v1', max_episode_steps=500)
    rng = np.random.default_rng(args.seed + 100000)
    episode = 0
    obs, _ = env.reset(seed=args.seed + episode)
    ep_return = 0.0
    completed_returns = []
    action_digest = hashlib.sha256()
    angles_digest = hashlib.sha256()
    rollout_seconds = gradient_seconds = update_seconds = 0.0
    train_started = time.perf_counter()
    with (args.output/'training.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['update', 'steps', 'episodes', 'last_episode_return', 'gradient_norm',
                         'rollout_seconds', 'gradient_seconds', 'update_seconds', 'elapsed_seconds'])
        for update in range(args.steps // args.batch_size):
            t0 = time.perf_counter()
            trajectory, rewards, dones = [], [], []
            for _ in range(args.batch_size):
                angles = pnp.array(np.arctan(obs), requires_grad=False)
                z = float(score(theta, angles))
                if not np.isfinite(z) or abs(z) > 1 + 1e-9:
                    raise FloatingPointError('Invalid parity expectation')
                action = int(rng.random() >= np.clip((1 + z) / 2, 0, 1))
                action_digest.update(bytes([action]))
                angles_digest.update(np.asarray(angles, dtype='<f8').tobytes())
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                trajectory.append((angles, action))
                rewards.append(float(reward))
                dones.append(done)
                ep_return += float(reward)
                if done:
                    completed_returns.append(ep_return)
                    episode += 1
                    obs, _ = env.reset(seed=args.seed + episode)
                    ep_return = 0.0
            rollout = time.perf_counter() - t0
            t0 = time.perf_counter()
            returns = np.zeros(args.batch_size)
            future = 0.0
            for t in reversed(range(args.batch_size)):
                future = rewards[t] + 0.99 * future * (not dones[t])
                returns[t] = future
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
            grad = np.zeros_like(theta0)
            for (angles, action), advantage in zip(trajectory, advantages):
                grad += np.asarray(gradient(theta, angles, action, float(advantage)))
            grad /= args.batch_size
            grad_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            norm = float(np.linalg.norm(grad))
            if not np.isfinite(grad).all():
                raise FloatingPointError('Nonfinite training gradient')
            grad *= min(1.0, 5.0/(norm+1e-12))
            theta = pnp.array(np.asarray(theta) - 0.02*grad, requires_grad=True)
            update_time = time.perf_counter() - t0
            rollout_seconds += rollout
            gradient_seconds += grad_time
            update_seconds += update_time
            writer.writerow([update, (update+1)*args.batch_size, episode,
                             completed_returns[-1] if completed_returns else '', norm,
                             rollout, grad_time, update_time, time.perf_counter()-train_started])
            f.flush()
    training_seconds = time.perf_counter() - train_started
    env.close()
    final_returns = evaluate(theta)
    save_json(args.output/'parameters.json', dict(initial=theta0.tolist(), final=np.asarray(theta).tolist()))
    save_json(args.output/'summary.json', dict(status='complete', seed=args.seed, threads=args.threads,
              training_steps=args.steps, updates=args.steps//args.batch_size,
              training_seconds=training_seconds, seconds_per_step=training_seconds/args.steps,
              rollout_seconds=rollout_seconds, gradient_seconds=gradient_seconds, update_seconds=update_seconds,
              fixed_probe_seconds=probe_seconds, fixed_probe_steps=args.probe_steps,
              initial_evaluation_returns=initial_returns, final_evaluation_returns=final_returns,
              initial_mean_return=float(np.mean(initial_returns)), final_mean_return=float(np.mean(final_returns)),
              completed_training_returns=completed_returns, partial_episode_return=ep_return,
              action_sha256=action_digest.hexdigest(), observation_sha256=angles_digest.hexdigest(),
              setup_and_execution_seconds=time.perf_counter()-started))
    print(json.dumps({'status':'complete', 'threads':args.threads, 'seed':args.seed,
                      'training_seconds':training_seconds, 'final_mean_return':float(np.mean(final_returns))}), flush=True)


if __name__ == '__main__':
    main()
