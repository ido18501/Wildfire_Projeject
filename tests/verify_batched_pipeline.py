#!/usr/bin/env python3
"""
Verification Script for Batched Training Pipeline

This script verifies correctness at each step:
1. Batched forward solver (v2) matches sequential solver
2. Batched backward pass matches sequential backward pass
3. End-to-end gradient flow is correct (finite difference verification)
4. Training loop runs without errors

Run this BEFORE starting actual training!
"""

import numpy as np
import torch
import time
import sys



def verify_forward_solver():
    """Test 1: Verify batched forward solver v2."""
    print("="*70)
    print("TEST 1: Batched Forward Solver v2")
    print("="*70)
    
    from eikonal.forward.pytorch import EikonalProblem
    from eikonal.forward.numba_cpu import fast_sweeping_solve_numba
    from forward_solver_batched_v2 import fast_sweeping_solve_batched_v2, warmup_v2
    
    # Test parameters
    K = 4  # Batch size
    H, W = 64, 64
    
    print(f"\nConfiguration: K={K}, H={H}, W={W}")
    
    # Create varied G/B for each fire
    np.random.seed(42)
    G_batch = np.ones((K, 3, H, W), dtype=np.float32)
    B_batch = np.zeros((K, 2, H, W), dtype=np.float32)
    
    for k in range(K):
        # Different slowness per fire
        scale = 0.2 + 0.3 * (k / K)
        G_batch[k, 0] = scale  # g11
        G_batch[k, 1] = 0.0    # g12
        G_batch[k, 2] = scale  # g22
        
        # Different drift per fire
        B_batch[k, 0] = 0.05 * (k - K/2)
    
    source_points = [(np.random.randint(15, H-15), np.random.randint(15, W-15)) 
                     for _ in range(K)]
    
    # Warmup
    warmup_v2()
    
    # Run batched solver
    print("\n1. Running batched v2 solver...")
    t0 = time.perf_counter()
    T_batch = fast_sweeping_solve_batched_v2(G_batch, B_batch, source_points, max_iter=50)
    t_batched = time.perf_counter() - t0
    print(f"   Time: {t_batched*1000:.2f} ms")
    
    # Run sequential solver for reference
    print("\n2. Running sequential Numba solver (reference)...")
    G_torch = torch.from_numpy(G_batch)
    B_torch = torch.from_numpy(B_batch)
    
    T_ref = np.zeros((K, H, W), dtype=np.float32)
    t0 = time.perf_counter()
    for k in range(K):
        source_mask = torch.zeros(1, H, W, dtype=torch.bool)
        source_mask[0, source_points[k][0], source_points[k][1]] = True
        
        problem = EikonalProblem(G_torch[k:k+1], B_torch[k:k+1], source_mask)
        T, _ = fast_sweeping_solve_numba(problem, max_iter=50)
        T_ref[k] = T[0].numpy()
    t_seq = time.perf_counter() - t0
    print(f"   Time: {t_seq*1000:.2f} ms")
    
    # Compare
    print("\n3. Comparing results...")
    all_pass = True
    for k in range(K):
        valid_mask = (T_ref[k] < 1e5) & (T_batch[k] < 1e5)
        diff = np.abs(T_ref[k] - T_batch[k])
        max_diff = diff[valid_mask].max()
        mean_diff = diff[valid_mask].mean()
        
        status = "✓" if max_diff < 1.0 else "✗"
        if max_diff >= 1.0:
            all_pass = False
        print(f"   Fire {k}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} {status}")
    
    print(f"\nSpeedup: {t_seq/t_batched:.2f}x")
    print(f"\nTEST 1: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def verify_backward_pass():
    """Test 2: Verify batched backward pass."""
    print("\n" + "="*70)
    print("TEST 2: Batched Backward Pass")
    print("="*70)
    
    from eikonal.forward.pytorch import EikonalProblem
    from eikonal.forward.numba_cpu import fast_sweeping_solve_numba
    from eikonal.backward.implicit_diff import backward_pass
    from eikonal.backward.adjoint_batched import batched_backward_pass
    
    K = 4
    H, W = 64, 64
    
    print(f"\nConfiguration: K={K}, H={H}, W={W}")
    
    # Create varied G/B
    torch.manual_seed(42)
    G_batch = torch.ones(K, 3, H, W) * 0.35
    G_batch[:, 1] = 0
    
    for k in range(K):
        scale = 0.2 + 0.3 * (k / K)
        G_batch[k] *= scale
    
    B_batch = torch.zeros(K, 2, H, W)
    for k in range(K):
        B_batch[k, 0] = 0.05 * (k - K/2)
    
    np.random.seed(42)
    source_points = [(np.random.randint(15, H-15), np.random.randint(15, W-15)) 
                     for _ in range(K)]
    
    # Forward solve
    print("\n1. Forward solving...")
    T_batch = torch.zeros(K, H, W)
    for k in range(K):
        source_mask = torch.zeros(1, H, W, dtype=torch.bool)
        source_mask[0, source_points[k][0], source_points[k][1]] = True
        problem = EikonalProblem(G_batch[k:k+1], B_batch[k:k+1], source_mask)
        T, _ = fast_sweeping_solve_numba(problem, max_iter=50)
        T_batch[k] = T[0]
    
    # Create loss gradients
    print("2. Creating loss gradients...")
    dL_dT_batch = torch.zeros(K, H, W)
    for k in range(K):
        obs_mask = torch.rand(H, W) < 0.3
        obs_mask &= (T_batch[k] < 1e5)
        T_target = T_batch[k] + torch.randn(H, W) * 2
        dL_dT_batch[k][obs_mask] = (T_batch[k][obs_mask] - T_target[obs_mask])
    
    # Sequential backward pass (reference)
    print("\n3. Running sequential backward pass (reference)...")
    t0 = time.perf_counter()
    dL_dG_seq = torch.zeros(K, 3, H, W)
    dL_dB_seq = torch.zeros(K, 2, H, W)
    
    for k in range(K):
        source_mask = torch.zeros(1, H, W, dtype=torch.bool)
        source_mask[0, source_points[k][0], source_points[k][1]] = True
        problem = EikonalProblem(G_batch[k:k+1], B_batch[k:k+1], source_mask)
        dL_dG_k, dL_dB_k = backward_pass(T_batch[k:k+1], dL_dT_batch[k:k+1], problem)
        dL_dG_seq[k] = dL_dG_k[0]
        dL_dB_seq[k] = dL_dB_k[0]
    t_seq = time.perf_counter() - t0
    print(f"   Time: {t_seq*1000:.1f} ms")
    
    # Batched backward pass
    print("\n4. Running batched backward pass...")
    t0 = time.perf_counter()
    dL_dG_batch, dL_dB_batch = batched_backward_pass(
        T_batch, dL_dT_batch, G_batch, B_batch, source_points
    )
    t_batched = time.perf_counter() - t0
    print(f"   Time: {t_batched*1000:.1f} ms")
    
    # Compare
    print("\n5. Comparing results...")
    all_pass = True
    for k in range(K):
        diff_G = (dL_dG_batch[k] - dL_dG_seq[k]).abs().max().item()
        diff_B = (dL_dB_batch[k] - dL_dB_seq[k]).abs().max().item()
        
        status_G = "✓" if diff_G < 1e-5 else "✗"
        status_B = "✓" if diff_B < 1e-5 else "✗"
        
        if diff_G >= 1e-5 or diff_B >= 1e-5:
            all_pass = False
        
        print(f"   Fire {k}: dL/dG diff={diff_G:.2e} {status_G}, dL/dB diff={diff_B:.2e} {status_B}")
    
    print(f"\nSpeedup: {t_seq/t_batched:.2f}x")
    print(f"\nTEST 2: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def verify_gradient_flow():
    """Test 3: Verify end-to-end gradient flow via finite differences."""
    print("\n" + "="*70)
    print("TEST 3: End-to-End Gradient Flow (Finite Differences)")
    print("="*70)
    
    from eikonal.forward.pytorch import EikonalProblem
    from eikonal.forward.numba_cpu import fast_sweeping_solve_numba
    from eikonal.backward.adjoint_batched import batched_backward_pass
    from batched_training import MetricNetwork
    
    # Small network for testing
    model = MetricNetwork(
        spatial_channels=4,
        global_features=3,
        cnn_hidden=16,
        cnn_layers=2,
        mlp_hidden=8,
        mlp_layers=2,
    )
    
    # Single fire for simplicity
    H, W = 32, 32
    C_spatial = 4
    C_global = 3
    
    torch.manual_seed(42)
    spatial = torch.randn(1, C_spatial, H, W)
    global_feat = torch.randn(1, C_global)
    source_point = [(16, 16)]
    
    print(f"\nConfiguration: H={H}, W={W}, testing one fire")
    
    # Forward through network
    G, B, _, _ = model(spatial, global_feat)
    
    # Forward solve
    source_mask = torch.zeros(1, H, W, dtype=torch.bool)
    source_mask[0, 16, 16] = True
    problem = EikonalProblem(G.detach(), B.detach(), source_mask)
    T, _ = fast_sweeping_solve_numba(problem, max_iter=50)
    
    # Create simple loss: L = T[20, 20]
    target_i, target_j = 20, 20
    dL_dT = torch.zeros_like(T)
    dL_dT[0, target_i, target_j] = 1.0
    
    loss_value = T[0, target_i, target_j].item()
    print(f"\nLoss = T[{target_i}, {target_j}] = {loss_value:.4f}")
    
    # Backward pass
    print("\n1. Computing gradients via implicit differentiation...")
    dL_dG, dL_dB = batched_backward_pass(T, dL_dT, G.detach(), B.detach(), source_point)
    
    # Backprop to network parameters
    model.zero_grad()
    G.backward(dL_dG, retain_graph=True)
    B.backward(dL_dB)
    
    # Get gradient for one parameter
    param = list(model.parameters())[0]  # First conv layer weight
    computed_grad = param.grad[0, 0, 0, 0].item()
    
    print(f"   Computed gradient (first param): {computed_grad:.6e}")
    
    # Finite difference verification
    print("\n2. Verifying with finite differences...")
    eps = 1e-4
    
    with torch.no_grad():
        # Perturb +
        param.data[0, 0, 0, 0] += eps
        G_plus, B_plus, _, _ = model(spatial, global_feat)
        problem_plus = EikonalProblem(G_plus, B_plus, source_mask)
        T_plus, _ = fast_sweeping_solve_numba(problem_plus, max_iter=50)
        loss_plus = T_plus[0, target_i, target_j].item()
        
        # Perturb -
        param.data[0, 0, 0, 0] -= 2 * eps
        G_minus, B_minus, _, _ = model(spatial, global_feat)
        problem_minus = EikonalProblem(G_minus, B_minus, source_mask)
        T_minus, _ = fast_sweeping_solve_numba(problem_minus, max_iter=50)
        loss_minus = T_minus[0, target_i, target_j].item()
        
        # Restore
        param.data[0, 0, 0, 0] += eps
    
    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    print(f"   Finite difference gradient: {fd_grad:.6e}")
    
    # Compare
    if abs(computed_grad) > 1e-10:
        rel_err = abs(fd_grad - computed_grad) / abs(fd_grad)
    else:
        rel_err = abs(fd_grad - computed_grad)
    
    status = "✓" if rel_err < 0.1 else "✗"
    print(f"\n   Relative error: {rel_err:.2e} {status}")
    
    all_pass = rel_err < 0.1
    print(f"\nTEST 3: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def verify_training_step():
    """Test 4: Verify training step runs without errors."""
    print("\n" + "="*70)
    print("TEST 4: Training Step Verification")
    print("="*70)
    
    from batched_training import (
        MetricNetwork, BatchedPatchDataset, 
        BatchedCrossSceneTrainer, collate_batch
    )
    
    # Create minimal synthetic data
    n_fires = 8
    H, W = 64, 64
    C_spatial = 8
    C_global = 6
    
    print(f"\nCreating {n_fires} synthetic fires...")
    
    fire_data = []
    for i in range(n_fires):
        spatial = np.random.randn(C_spatial, H, W).astype(np.float32)
        global_feat = np.random.randn(C_global).astype(np.float32)
        source_i, source_j = np.random.randint(20, H-20), np.random.randint(20, W-20)
        
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - source_j)**2 + (Y - source_i)**2)
        T_obs = dist.astype(np.float32) + np.random.randn(H, W).astype(np.float32) * 0.5
        T_obs[source_i, source_j] = 0
        
        obs_mask = np.random.rand(H, W) < 0.5
        obs_mask[source_i, source_j] = False
        
        fire_data.append({
            'spatial_covariates': spatial,
            'global_features': global_feat,
            'arrival_times': T_obs,
            'obs_mask': obs_mask,
            'source_point': (source_i, source_j),
            'fire_id': f'fire_{i:03d}',
        })
    
    # Create dataset
    dataset = BatchedPatchDataset(fire_data[:6], patch_size=64)
    val_dataset = BatchedPatchDataset(fire_data[6:], patch_size=64)
    val_dataset.spatial_mean = dataset.spatial_mean
    val_dataset.spatial_std = dataset.spatial_std
    val_dataset.global_mean = dataset.global_mean
    val_dataset.global_std = dataset.global_std
    
    # Create model
    model = MetricNetwork(
        spatial_channels=C_spatial,
        global_features=C_global,
        cnn_hidden=32,
        cnn_layers=3,
        mlp_hidden=16,
        mlp_layers=2,
    )
    
    # Quick test that model returns 4 values
    test_spatial = torch.randn(2, C_spatial, H, W)
    test_global = torch.randn(2, C_global)
    outputs = model(test_spatial, test_global)
    if len(outputs) != 4:
        print(f"   FAILED: Model returns {len(outputs)} values, expected 4")
        return False
    print(f"   Model returns {len(outputs)} values (G, B, G_local, B_local) ✓")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create trainer
    trainer = BatchedCrossSceneTrainer(
        model=model,
        train_dataset=dataset,
        val_dataset=val_dataset,
        batch_size=2,
        lr=1e-3,
        lambda_tv=0.001,
        device=device,
        use_cuda_forward=(device == 'cuda'),
    )
    
    # Run one training step
    print("\n1. Running one training step...")
    try:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_batch)
        batch = next(iter(loader))
        
        t0 = time.perf_counter()
        losses = trainer.train_step(batch)
        t_step = time.perf_counter() - t0
        
        print(f"   Data loss: {losses['data_loss']:.4f}")
        print(f"   TV loss: {losses['tv_loss']:.6f}")
        print(f"   Total loss: {losses['total_loss']:.4f}")
        print(f"   Time: {t_step*1000:.1f} ms")
        step_pass = True
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        step_pass = False
    
    # Verify gradients are non-zero
    print("\n2. Checking gradients...")
    grad_nonzero = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.abs().mean().item()
            if grad_norm > 1e-10:
                grad_nonzero = True
                print(f"   {name}: grad norm = {grad_norm:.2e}")
                break
    
    if grad_nonzero:
        print("   ✓ Gradients are non-zero")
    else:
        print("   ✗ Gradients are zero or None!")
    
    # Run validation
    print("\n3. Running validation...")
    try:
        val_metrics = trainer.validate()
        print(f"   Validation correlation: {val_metrics.get('val_corr', 0):.4f}")
        val_pass = True
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        val_pass = False
    
    all_pass = step_pass and grad_nonzero and val_pass
    print(f"\nTEST 4: {'PASSED' if all_pass else 'FAILED'}")
    return all_pass


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("BATCHED TRAINING PIPELINE VERIFICATION")
    print("="*70)
    print("\nRunning all verification tests...")
    
    results = {}
    
    # Test 1: Forward solver
    results['forward_solver'] = verify_forward_solver()
    
    # Test 2: Backward pass
    results['backward_pass'] = verify_backward_pass()
    
    # Test 3: Gradient flow
    results['gradient_flow'] = verify_gradient_flow()
    
    # Test 4: Training step
    results['training_step'] = verify_training_step()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    print("\n" + "-"*70)
    if all_pass:
        print("ALL TESTS PASSED! Pipeline is ready for training.")
    else:
        print("SOME TESTS FAILED! Please fix issues before training.")
    print("="*70)
    
    return all_pass


if __name__ == "__main__":
    run_all_tests()
