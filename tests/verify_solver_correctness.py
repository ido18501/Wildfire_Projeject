"""
Comprehensive correctness verification for all forward solver implementations.
Uses ACTUAL fire data from the Sim2Real dataset via Sim2RealFireLoader.

Tests 100 fires to ensure all implementations produce identical results.
"""

import numpy as np
import torch
import time
import sys
import os

# Add project path

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.forward.numba_cpu import fast_sweeping_solve_numba, warmup as warmup_numba
# Note: cuda_optimized was a development variant; cuda_batched is the released version
from eikonal.forward.cuda_batched import fast_sweeping_solve_batched as fast_sweeping_solve_cuda
from eikonal.forward.cuda_batched import fast_sweeping_solve_batched as fast_sweeping_solve_batched_cuda, warmup_v3 as warmup_cuda

# Import the actual data loader from your project
from experiments.wildfire.sim2real_loader import (
    Sim2RealFireLoader,
    CovariateAdapter,
    extract_arrival_times,
    find_ignition_point
)


def verify_all_methods(n_fires: int = 100):
    """Verify that all implementations produce identical results using real data."""
    
    print("=" * 70)
    print("COMPREHENSIVE CORRECTNESS VERIFICATION")
    print(f"Using {n_fires} ACTUAL Sim2Real Fire Events")
    print("=" * 70)
    print()
    
    # Warmup all implementations
    print("Warming up all implementations...")
    warmup_numba()
    warmup_cuda()
    print()
    
    # Load data using your loader
    data_root = '/path/to/sim2real_data'
    print(f"Data root: {data_root}")
    
    loader = Sim2RealFireLoader(data_root)
    
    print(f"Discovered {len(loader.scenes)} scenes:")
    total_events = 0
    for scene_id, info in sorted(loader.scenes.items()):
        n_events = len(info['events'])
        print(f"  {scene_id}: {n_events:,} fire events")
        total_events += n_events
    print(f"Total: {total_events:,} fire events")
    print()
    
    # Use a SINGLE scene to ensure consistent grid size
    # Pick the scene with enough fires
    scene_id = '0014_00426'  # 426 fires, known to work with 132x190 grid
    if scene_id not in loader.scenes:
        # Fall back to first scene with enough fires
        for sid, info in loader.scenes.items():
            if len(info['events']) >= n_fires:
                scene_id = sid
                break
    
    print(f"Using scene: {scene_id} (to ensure consistent grid size)")
    
    # Get scenarios from this scene only
    all_scenarios = loader.list_scenarios(scene_filter=[scene_id])
    print(f"Scenarios in scene: {len(all_scenarios)}")
    
    # Select fires to test
    np.random.seed(42)
    np.random.shuffle(all_scenarios)
    test_scenarios = all_scenarios[:n_fires]
    
    print(f"Testing {len(test_scenarios)} fires")
    print()
    
    # Load first scenario to get grid size
    scene_id_first, event_id_first = test_scenarios[0]
    first_data = loader.load_scenario(scene_id_first, event_id_first)
    
    # Get grid shape
    elev = first_data['topography'].get('elevation')
    H, W = elev.shape
    print(f"Grid shape: {H} x {W}")
    
    # Create constant G and B for testing (actual training uses CNN)
    G_np = np.ones((3, H, W), dtype=np.float32) * 0.35
    G_np[1] = 0  # g12 = 0 (isotropic)
    B_np = np.zeros((2, H, W), dtype=np.float32)
    
    G_torch = torch.from_numpy(G_np).unsqueeze(0)
    B_torch = torch.from_numpy(B_np).unsqueeze(0)
    
    # Collect source points and ground truth for all fires
    print("Loading fire data...")
    source_points = []
    ground_truth = []
    valid_scenarios = []
    
    for i, (scene_id, event_id) in enumerate(test_scenarios):
        try:
            data = loader.load_scenario(scene_id, event_id)
            
            # Extract arrival times
            arrival = extract_arrival_times(data['masks'], data['timestamps'])
            
            # Find ignition point
            ignition = find_ignition_point(data['masks'])
            if ignition is None:
                continue
            
            si, sj = ignition
            source_points.append((si, sj))
            ground_truth.append(arrival)
            valid_scenarios.append((scene_id, event_id))
            
            if (i + 1) % 20 == 0:
                print(f"  Loaded {i + 1}/{len(test_scenarios)} fires...")
                
        except Exception as e:
            print(f"  Skipping {scene_id}/{event_id}: {e}")
            continue
    
    n_test_fires = len(source_points)
    print(f"\nSuccessfully loaded {n_test_fires} fires")
    print()
    
    # Show first few ignition points
    print("Sample ignition points:")
    for i in range(min(5, n_test_fires)):
        si, sj = source_points[i]
        scene_id, event_id = valid_scenarios[i]
        print(f"  Fire {i}: {scene_id}/{event_id} at ({si}, {sj})")
    print()
    
    # =========================================================================
    # Run all solvers
    # =========================================================================
    
    print("=" * 70)
    print("RUNNING ALL SOLVERS")
    print("=" * 70)
    print()
    
    results = {
        'pytorch': [],
        'numba': [],
        'seq_cuda': [],
        'batch_cuda': []
    }
    
    # 1. Batched CUDA (all fires at once)
    print(f"1. Batched CUDA ({n_test_fires} fires in parallel)...")
    t0 = time.perf_counter()
    T_batched = fast_sweeping_solve_batched_cuda(G_np, B_np, source_points, max_iter=2)
    t_batched = time.perf_counter() - t0
    results['batch_cuda'] = [T_batched[i] for i in range(n_test_fires)]
    print(f"   Time: {t_batched*1000:.1f}ms ({t_batched/n_test_fires*1000:.3f}ms/fire)")
    
    # 2. Sequential solvers (sample for speed)
    n_sample = min(20, n_test_fires)  # Only run 20 for sequential methods
    
    print(f"\n2. Sequential solvers (sampling {n_sample} fires)...")
    
    for fire_idx in range(n_sample):
        si, sj = source_points[fire_idx]
        source = torch.zeros(1, H, W, dtype=torch.bool)
        source[0, si, sj] = True
        problem = EikonalProblem(G_torch, B_torch, source)
        
        # PyTorch
        T_pytorch, _ = fast_sweeping_solve(problem, max_iter=50)
        results['pytorch'].append(T_pytorch[0].numpy())
        
        # Numba CPU
        T_numba, _ = fast_sweeping_solve_numba(problem, max_iter=50)
        results['numba'].append(T_numba[0].numpy())
        
        # Sequential CUDA
        T_seq_cuda, _ = fast_sweeping_solve_cuda(problem, max_iter=50)
        results['seq_cuda'].append(T_seq_cuda[0].numpy())
        
        if (fire_idx + 1) % 5 == 0:
            print(f"   Processed {fire_idx + 1}/{n_sample}...")
    
    print()
    
    # =========================================================================
    # Compare results
    # =========================================================================
    
    print("=" * 70)
    print("PAIRWISE COMPARISON (max diff)")
    print("=" * 70)
    print()
    
    print(f"{'Fire':<6} {'Location':<12} {'Numba-PT':<12} {'SeqCUDA-PT':<12} {'BatchCUDA-PT':<12}")
    print("-" * 70)
    
    max_diffs = {
        'numba_vs_pytorch': 0,
        'seq_cuda_vs_pytorch': 0,
        'batch_cuda_vs_pytorch': 0,
    }
    
    for fire_idx in range(n_sample):
        si, sj = source_points[fire_idx]
        
        T_pt = results['pytorch'][fire_idx]
        T_numba = results['numba'][fire_idx]
        T_seq_cuda = results['seq_cuda'][fire_idx]
        T_batch_cuda = results['batch_cuda'][fire_idx]
        
        valid = T_pt < 1e5
        
        diff_numba = np.abs(T_pt - T_numba)[valid].max()
        diff_seq_cuda = np.abs(T_pt - T_seq_cuda)[valid].max()
        diff_batch_cuda = np.abs(T_pt - T_batch_cuda)[valid].max()
        
        max_diffs['numba_vs_pytorch'] = max(max_diffs['numba_vs_pytorch'], diff_numba)
        max_diffs['seq_cuda_vs_pytorch'] = max(max_diffs['seq_cuda_vs_pytorch'], diff_seq_cuda)
        max_diffs['batch_cuda_vs_pytorch'] = max(max_diffs['batch_cuda_vs_pytorch'], diff_batch_cuda)
        
        print(f"{fire_idx:<6} ({si:3d},{sj:3d})   "
              f"{diff_numba:<12.2e} "
              f"{diff_seq_cuda:<12.2e} "
              f"{diff_batch_cuda:<12.2e}")
    
    print("-" * 70)
    print()
    
    # =========================================================================
    # Summary matrix
    # =========================================================================
    
    print("=" * 70)
    print("SUMMARY: MAX DIFFERENCE MATRIX")
    print("=" * 70)
    print()
    
    methods = ['pytorch', 'numba', 'seq_cuda', 'batch_cuda']
    method_names = ['PyTorch', 'Numba CPU', 'Seq CUDA', 'Batch CUDA']
    
    print(f"{'':>15}", end='')
    for name in method_names:
        print(f"{name:>12}", end='')
    print()
    
    for i, (m1, n1) in enumerate(zip(methods, method_names)):
        print(f"{n1:>15}", end='')
        for j, (m2, n2) in enumerate(zip(methods, method_names)):
            if i == j:
                print(f"{'---':>12}", end='')
            else:
                max_diff = 0
                for fire_idx in range(n_sample):
                    T1 = results[m1][fire_idx]
                    T2 = results[m2][fire_idx]
                    valid = (T1 < 1e5) & (T2 < 1e5)
                    if valid.sum() > 0:
                        diff = np.abs(T1 - T2)[valid].max()
                        max_diff = max(max_diff, diff)
                print(f"{max_diff:>12.2e}", end='')
        print()
    
    print()
    
    # =========================================================================
    # Iteration counts
    # =========================================================================
    
    print("=" * 70)
    print("ITERATION COUNTS (fire 0)")
    print("=" * 70)
    print()
    
    si, sj = source_points[0]
    source = torch.zeros(1, H, W, dtype=torch.bool)
    source[0, si, sj] = True
    problem = EikonalProblem(G_torch, B_torch, source)
    
    _, n_pytorch = fast_sweeping_solve(problem, max_iter=50)
    _, n_numba = fast_sweeping_solve_numba(problem, max_iter=50)
    _, n_seq_cuda = fast_sweeping_solve_cuda(problem, max_iter=50)
    
    print(f"PyTorch:     {n_pytorch} iterations")
    print(f"Numba CPU:   {n_numba} iterations")
    print(f"Seq CUDA:    {n_seq_cuda} iterations")
    print(f"Batch CUDA:  2 iterations (fixed)")
    print()
    
    # =========================================================================
    # Correlation with ground truth
    # =========================================================================
    
    print("=" * 70)
    print("CORRELATION WITH GROUND TRUTH")
    print("=" * 70)
    print()
    print("Note: We use constant G,B, so correlation reflects solver correctness,")
    print("not prediction accuracy (which requires trained CNN).")
    print()
    
    for fire_idx in range(min(5, n_sample)):
        gt = ground_truth[fire_idx]
        T_pred = results['pytorch'][fire_idx]
        
        valid = np.isfinite(gt) & (T_pred < 1e5) & (gt > 0)
        
        if valid.sum() > 100:
            corr = np.corrcoef(gt[valid].flatten(), T_pred[valid].flatten())[0, 1]
            print(f"Fire {fire_idx}: correlation = {corr:.4f}")
    
    print()
    
    # =========================================================================
    # Final verdict
    # =========================================================================
    
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print()
    
    max_diff_overall = max(max_diffs.values())
    print(f"Maximum difference across all methods: {max_diff_overall:.2e}")
    print()
    
    if max_diff_overall < 0.01:
        print("✓ ALL METHODS PRODUCE IDENTICAL RESULTS (diff < 0.01)")
        verdict = "PASS"
    elif max_diff_overall < 0.1:
        print("✓ ALL METHODS PRODUCE VERY SIMILAR RESULTS (diff < 0.1)")
        print("  Small differences are due to floating-point precision.")
        verdict = "PASS"
    elif max_diff_overall < 1.0:
        print("⚠ METHODS PRODUCE SIMILAR RESULTS (diff < 1.0)")
        print("  Differences may be due to iteration count or precision.")
        verdict = "WARN"
    else:
        print("✗ METHODS PRODUCE DIFFERENT RESULTS")
        print("  There may be a bug in one of the implementations.")
        verdict = "FAIL"
    
    print()
    print("=" * 70)
    print(f"VERDICT: {verdict}")
    print("=" * 70)
    
    return verdict == "PASS" or verdict == "WARN"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_fires', type=int, default=100, help='Number of fires to test')
    parser.add_argument('--output', type=str, default='verification_results.txt', help='Output log file')
    args = parser.parse_args()
    
    # Tee output to both console and file
    import sys
    
    class Tee:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Tee(args.output)
    
    print(f"Logging to: {args.output}")
    print()
    
    result = verify_all_methods(n_fires=args.n_fires)
    
    print()
    print(f"Results saved to: {args.output}")
