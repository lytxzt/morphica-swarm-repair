"""Re-run algorithms at N1000 d=500, saving final positions as .npy for E6 topology analysis."""
import sys, numpy as np, random, os, json

sys.argv = ['e6', '--num-agents', '1000', '--destroyed', '500']
from Configurations import *
from Environment import Environment
from Swarm import Swarm
import Utils

N_CASES = 20
algorithms = {
    'MORPHICA': None,
    'ML-DAGL': 8,   # mode 8
    'CEN': 2,        # mode 2
    'CR-MGC': 6,     # mode 6
    'DEMD': 7,       # mode 7
}

for alg_name, mode in algorithms.items():
    print(f"\n{'='*60}")
    print(f"Running {alg_name} ({N_CASES} cases)...")
    results = []
    
    for ci in range(N_CASES):
        try:
            seed_val = config_random_seed[ci] + 1718
            np.random.seed(seed_val)
            random.seed(seed_val)
            
            env = Environment(); env.reset()
            algo_mode = 14 if alg_name == 'MORPHICA' else mode
            swarm = Swarm(algorithm_mode=algo_mode)
            if alg_name == 'MORPHICA':
                swarm.reset()
            else:
                swarm.reset()
            
            destroy_num, destroy_list = env.stochastic_destroy(mode=2, num_of_destroyed=500)
            swarm.destroy_happens(destroy_list, env.environment_positions)
            swarm.make_remain_positions()
            
            step, final_positions, num = swarm.solve()
            converged = (step < int(0.8 * config_width))
            
            out_dir = f"Logs/e6_positions/{alg_name}"
            os.makedirs(out_dir, exist_ok=True)
            np.save(f"{out_dir}/case{ci}.npy", final_positions)
            
            n = len(final_positions)
            G = np.matmul(final_positions, final_positions.T)
            H = np.tile(np.diag(G), (n, 1))
            D = np.sqrt(H + H.T - 2 * G)
            A = (D <= config_communication_range).astype(float)
            np.fill_diagonal(A, 0)
            deg = A.sum(axis=0)
            
            r = {'case': ci, 'step': step, 'converged': converged,
                 'n_agents': n, 'num_clusters': num,
                 'avg_deg': float(np.mean(deg)), 'min_deg': float(np.min(deg))}
            results.append(r)
            status = "✓" if converged else "✗"
            print(f"  case {ci:3d}: step={step:5d} {status}  avg_deg={r['avg_deg']:.2f}  n={n}")
        except Exception as e:
            print(f"  case {ci:3d}: ERROR {e}")
            import traceback; traceback.print_exc()
    
    with open(f"Logs/e6_positions/{alg_name}_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    valid = [r for r in results if r['converged']]
    if valid:
        ad = np.mean([r['avg_deg'] for r in valid])
        md = np.mean([r['min_deg'] for r in valid])
        print(f"  Avg of {len(valid)} converged: avg_deg={ad:.2f}, min_deg={md:.2f}")

print("\nDone!")
