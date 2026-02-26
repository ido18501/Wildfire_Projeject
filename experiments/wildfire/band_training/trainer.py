"""
Band-MSE Trainer
=================

SceneTrainer class implementing band-decomposed MSE training with
teacher forcing for Sim2Real-Fire wildfire propagation.

Core loop per fire:
  1. CNN forward (once) → shared metric field G, B
  2. Generate K overlapping bands; build teacher-forced source masks
  3. Batched forward solve (CUDA single / Numba prange / sequential)
  4. MSE loss per band in relative coordinates
  5. Batched backward pass (Numba prange over K bands)
  6. Average gradients across bands → single optimizer step

Author: Barak Gahtan
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn

# Project imports

from eikonal.forward.pytorch import EikonalProblem, fast_sweeping_solve
from eikonal.forward.numba_cpu import (
    fast_sweeping_solve_numba, warmup as warmup_numba,
    _fast_sweeping_iteration, _compute_max_change)
from eikonal.backward.implicit_diff import (
    backward_pass, identify_active_stencils, compute_parameter_gradients)

try:
    from eikonal.backward.adjoint_numba import solve_adjoint_numba, HAVE_NUMBA
except ImportError:
    HAVE_NUMBA = False

# Sparse adjoint for band training (processes only active pixels)
_HAVE_SPARSE_ADJOINT = False
solve_adjoint_numba_sparse = None
try:
    from eikonal.backward.adjoint_sparse import (
        solve_adjoint_numba_sparse, warmup_sparse_adjoint)
    _HAVE_SPARSE_ADJOINT = True
except ImportError:
    def warmup_sparse_adjoint():
        pass

try:
    from numba import jit, prange
    _HAVE_NUMBA_JIT = True
except ImportError:
    _HAVE_NUMBA_JIT = False

try:
    from eikonal.forward.cuda_batched import (
        fast_sweeping_solve_batched, warmup_v3 as warmup_batched)
    HAVE_BATCHED_CUDA = True
except ImportError:
    HAVE_BATCHED_CUDA = False
    fast_sweeping_solve_batched = None
    warmup_batched = None

from experiments.wildfire.sim2real_loader import (
    Sim2RealFireLoader, CovariateAdapter,
    extract_arrival_times, find_ignition_point)
from eikonal.nn.encoder import FlexibleCovariateEncoder
from eikonal.nn.regularization import compute_tv_regularization
from eikonal.nn.config import DEVICE, TVMetricType

from band_mse_config import (
    ExperimentConfig, compute_all_metrics, aggregate_metrics)


# ============================================================================
# Solver configuration
# ============================================================================

# Force sequential Numba — Numba CUDA JIT segfaults on this setup
# (matching workaround from experiment_suite_band_mse.py)
HAVE_BATCHED_CUDA = False
warmup_batched = None
fast_sweeping_solve_batched = None

# Single-fire CUDA optimized solver — enabled on demand via enable_cuda_single()
HAVE_CUDA_SINGLE = False
_cuda_single_solver = None


def enable_cuda_single():
    """Import, warmup, and enable the single-fire CUDA optimized solver.

    Call only after confirming (e.g. via subprocess probe in run_band_mse.py)
    that the Numba CUDA JIT compiles without segfaulting on this machine.
    """
    global HAVE_CUDA_SINGLE, _cuda_single_solver
    from eikonal.forward.cuda_batched import (
        fast_sweeping_solve_batched as fast_sweeping_solve_cuda, warmup_v3 as warmup_cuda)
    warmup_cuda()
    _cuda_single_solver = fast_sweeping_solve_cuda
    HAVE_CUDA_SINGLE = True


# ============================================================================
# Parallel CPU band solver (Numba prange over K bands)
# ============================================================================

if _HAVE_NUMBA_JIT:
    @jit(nopython=True, parallel=True)
    def _solve_bands_parallel(G, B, source_batch, max_iter, tol):
        """Solve K band eikonal problems in parallel using Numba prange.

        All K bands share the same G, B (read-only).  Each thread
        handles one fire independently — no synchronization needed.

        Args:
            G: (3, H, W) shared metric tensor (float32)
            B: (2, H, W) shared drift field  (float32)
            source_batch: (K, H, W) boolean source masks
            max_iter: max sweeping iterations per band
            tol: convergence tolerance
        Returns:
            T_batch: (K, H, W) arrival times (float32)
        """
        K = source_batch.shape[0]
        M = source_batch.shape[1]
        N = source_batch.shape[2]
        T_batch = np.empty((K, M, N), dtype=np.float32)

        for k in prange(K):
            # Initialize T
            for i in range(M):
                for j in range(N):
                    if source_batch[k, i, j]:
                        T_batch[k, i, j] = 0.0
                    else:
                        T_batch[k, i, j] = 1e6

            # Fast sweeping iterations until convergence
            for iteration in range(max_iter):
                T_prev = T_batch[k].copy()
                _fast_sweeping_iteration(T_batch[k], G, B)
                mc = _compute_max_change(T_batch[k], T_prev)
                if mc <= tol:
                    break

        return T_batch

    def warmup_parallel_bands():
        """Compile the parallel band solver (tiny dummy problem)."""
        G = np.ones((3, 20, 20), dtype=np.float32) * 0.35
        G[1] = 0
        B = np.zeros((2, 20, 20), dtype=np.float32)
        src = np.zeros((2, 20, 20), dtype=np.bool_)
        src[0, 10, 10] = True
        src[1, 10, 10] = True
        _solve_bands_parallel(G, B, src, 2, 1e-6)

else:
    _solve_bands_parallel = None

    def warmup_parallel_bands():
        pass


print(f"[Band-MSE] HAVE_NUMBA: {HAVE_NUMBA}, "
      f"HAVE_BATCHED_CUDA: {HAVE_BATCHED_CUDA}, "
      f"HAVE_CUDA_SINGLE: {HAVE_CUDA_SINGLE}, "
      f"parallel_bands: {_solve_bands_parallel is not None}, "
      f"sparse_adjoint: {_HAVE_SPARSE_ADJOINT}")


# =============================================================================
# SceneTrainer
# =============================================================================

class SceneTrainer:
    """Trainer for a single scene with batched band-decomposed MSE."""

    def __init__(self, config: ExperimentConfig, loader):
        self.config = config
        self.loader = loader
        self.device = torch.device(config.device)
        self.adapter = CovariateAdapter()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.H = self.W = None

    def setup_model(self, H, W, n_spatial_features=6):
        self.H, self.W = H, W
        self.model = FlexibleCovariateEncoder(
            n_spatial_features=n_spatial_features,
            n_global_features=5,
            hidden_channels=self.config.hidden_channels,
            num_cnn_layers=self.config.num_cnn_layers,
            mlp_hidden=self.config.mlp_hidden,
            mlp_num_layers=self.config.mlp_num_layers,
            eps=self.config.eps,
            max_drift=self.config.max_drift,
            max_eigenvalue=self.config.max_eigenvalue,
            local_scale=self.config.local_scale,
        ).to(self.device)

        with torch.no_grad():
            ev = self.config.initial_eigenvalue
            self.model.global_mlp[-1].bias.copy_(
                torch.tensor([ev, 0.0, ev, 0.0, 0.0], device=self.device))

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5,
            patience=self.config.scheduler_patience, min_lr=1e-6)

    # -----------------------------------------------------------------
    # Data preparation
    # -----------------------------------------------------------------

    def prepare_scenario(self, scene_id, event_id):
        """Load a fire scenario and prepare tensors for training."""
        try:
            data = self.loader.load_scenario(scene_id, event_id)
            spatial_feat, global_feat = self.adapter(data)
            arrival = extract_arrival_times(data['masks'], data['timestamps'])
            ignition = find_ignition_point(data['masks'])
            if ignition is None:
                return None
            H, W = arrival.shape
            si, sj = ignition
            source_mask = np.zeros((H, W), dtype=bool)
            source_mask[max(0, si - 1):min(H, si + 2),
                        max(0, sj - 1):min(W, sj + 2)] = True
            return {
                'spatial_feat': spatial_feat,
                'global_feat': global_feat,
                'arrival_times': arrival,
                'source_mask': source_mask,
                'valid_mask': np.isfinite(arrival),
                'ignition': (si, sj),
                'scene_id': scene_id,
                'event_id': event_id,
            }
        except Exception:
            return None

    # -----------------------------------------------------------------
    # Band generation
    # -----------------------------------------------------------------

    def generate_bands(self, max_time, overlapping=True):
        """Generate temporal bands as fractions of max_time.

        Training uses overlapping bands (stride < width) for gradient
        coverage.  Evaluation uses non-overlapping bands (stride = width)
        so each temporal region is scored exactly once.
        """
        width = self.config.band_width * max_time
        stride = (self.config.band_stride if overlapping
                  else self.config.band_width) * max_time
        bands = []
        t = 0.0
        while t < max_time - 1e-6:
            t_end = min(t + width, max_time)
            bands.append((t, t_end))
            t += stride
            if t_end >= max_time:
                break
        return bands

    # -----------------------------------------------------------------
    # Helpers: sources & batched forward solve
    # -----------------------------------------------------------------

    def _build_band_sources(self, scenario, bands, true_T, valid):
        """Build teacher-forced source masks for each band.

        Band 0 uses the ignition mask.  Band k>0 treats all GT pixels
        with T <= t_start as already-burned sources.
        """
        sources, indices = [], []
        for k, (t_start, t_end) in enumerate(bands):
            if t_start < 1e-6:
                src = torch.from_numpy(scenario['source_mask'])
            else:
                src = (true_T <= t_start) & valid
            if src.sum() >= 1:
                sources.append(src)
                indices.append(k)
        if not sources:
            return None, None
        return torch.stack(sources), indices

    def _solve_bands_forward(self, G_cpu, B_cpu, source_masks_K):
        """Forward-solve K band problems sharing the same metric.

        Priority:
          1. CUDA single-fire solver (GPU, sequential over K bands)
          2. Numba prange (CPU, parallel over K bands)
          3. Sequential Numba (CPU, one band at a time)

        Band solves use band_max_iter (default 20) instead of max_iter
        (50) because teacher-forced sources are large, so the wavefront
        only propagates one band-width and converges much faster.
        """
        K = source_masks_K.shape[0]
        H, W = source_masks_K.shape[1], source_masks_K.shape[2]
        band_iter = self.config.band_max_iter

        # Priority 1: CUDA single-fire solver (GPU-accelerated)
        if HAVE_CUDA_SINGLE:
            T_batch = torch.zeros(K, H, W)
            for k in range(K):
                prob = EikonalProblem(G_cpu, B_cpu, source_masks_K[k:k + 1])
                T_k, _ = _cuda_single_solver(prob, max_iter=band_iter)
                T_batch[k] = T_k[0]
            return T_batch

        # Priority 2: Numba prange — parallel over K bands on CPU
        if _solve_bands_parallel is not None:
            G_np = np.ascontiguousarray(
                G_cpu[0].numpy(), dtype=np.float32)
            B_np = np.ascontiguousarray(
                B_cpu[0].numpy(), dtype=np.float32)
            src_np = np.ascontiguousarray(source_masks_K.numpy())
            T_np = _solve_bands_parallel(G_np, B_np, src_np, band_iter, 1e-6)
            return torch.from_numpy(T_np).float()

        # Priority 3: sequential Numba (fallback)
        T_batch = torch.zeros(K, H, W)
        for k in range(K):
            prob = EikonalProblem(G_cpu, B_cpu, source_masks_K[k:k + 1])
            T_k, _ = fast_sweeping_solve_numba(prob, max_iter=band_iter)
            T_batch[k] = T_k[0]
        return T_batch

    # -----------------------------------------------------------------
    # Band-decomposed training step
    # -----------------------------------------------------------------

    def train_step(self, scenario):
        """One training step for a single fire with K band decomposition.

        Returns dict with loss, data_loss, reg_loss, n_bands or None on
        failure (NaN, too few pixels, etc.).
        """
        self.model.train()
        self.optimizer.zero_grad()

        # --- Step 1: CNN forward (once for all bands) ---
        spatial = scenario['spatial_feat'].unsqueeze(0).to(self.device)
        global_f = scenario['global_feat'].unsqueeze(0).to(self.device)
        G, B, G_local, B_local = self.model(spatial, global_f)

        if torch.isnan(G).any() or torch.isnan(B).any():
            return None

        G_cpu = G.detach().cpu()    # (1, 3, H, W)
        B_cpu = B.detach().cpu()    # (1, 2, H, W)
        true_T = torch.from_numpy(scenario['arrival_times']).float()
        valid = torch.from_numpy(scenario['valid_mask'])

        finite_valid = valid & torch.isfinite(true_T)
        if finite_valid.sum() < 10:
            return None
        max_time = true_T[finite_valid].max().item()
        if max_time < 1e-6:
            return None

        # --- Step 2: Generate bands and build source masks ---
        bands = self.generate_bands(max_time, overlapping=True)
        source_masks_K, valid_indices = self._build_band_sources(
            scenario, bands, true_T, valid)
        if source_masks_K is None:
            return None

        K_valid = source_masks_K.shape[0]
        bands_valid = [bands[k] for k in valid_indices]

        # --- Step 3: Batched forward solve ---
        T_batch = self._solve_bands_forward(G_cpu, B_cpu, source_masks_K)
        if torch.isnan(T_batch).any():
            return None

        # --- Step 4: Compute dL/dT per band (MSE in relative coords) ---
        H, W = true_T.shape
        dL_dT_batch = torch.zeros(K_valid, H, W)
        total_loss = 0.0
        n_loss_bands = 0

        for k_idx, (t_start, t_end) in enumerate(bands_valid):
            delta_t = t_end - t_start
            if delta_t < 1e-6:
                continue

            band_mask = (valid
                         & (true_T > t_start) & (true_T <= t_end)
                         & (T_batch[k_idx] < 1e5))
            n_band = band_mask.sum().item()
            if n_band < 5:
                continue

            gt_relative = true_T[band_mask] - t_start
            pred_vals = T_batch[k_idx][band_mask]
            residuals = pred_vals - gt_relative

            band_loss = 0.5 * (residuals ** 2).mean()
            if band_loss.item() > 1e6 or torch.isnan(band_loss):
                continue

            dL_dT_batch[k_idx][band_mask] = residuals / n_band
            total_loss += band_loss.item()
            n_loss_bands += 1

        if n_loss_bands == 0:
            return None

        # --- Step 5: Batched backward pass ---
        G_K = G_cpu.expand(K_valid, -1, -1, -1).contiguous()
        B_K = B_cpu.expand(K_valid, -1, -1, -1).contiguous()
        problem_K = EikonalProblem(G_K, B_K, source_masks_K)

        if HAVE_NUMBA:
            stencil_info = identify_active_stencils(T_batch, problem_K)
            if self.config.use_sparse_backward and _HAVE_SPARSE_ADJOINT:
                lam = solve_adjoint_numba_sparse(
                    stencil_info, dL_dT_batch, problem_K, T_batch)
            else:
                lam = solve_adjoint_numba(
                    stencil_info, dL_dT_batch, problem_K, T_batch)
            dL_dG_K, dL_dB_K = compute_parameter_gradients(
                stencil_info, lam, problem_K)
        else:
            dL_dG_K, dL_dB_K = backward_pass(
                T_batch, dL_dT_batch, problem_K)

        if torch.isnan(dL_dG_K).any() or torch.isnan(dL_dB_K).any():
            return None

        # --- Step 6: Average over bands, mask, update ---
        total_dL_dG = dL_dG_K.mean(dim=0, keepdim=True)   # (1, 3, H, W)
        total_dL_dB = dL_dB_K.mean(dim=0, keepdim=True)   # (1, 2, H, W)

        avg_data_loss = total_loss / n_loss_bands

        valid_tensor = torch.from_numpy(
            scenario['valid_mask']).float()[None, None, :, :]
        total_dL_dG = total_dL_dG * valid_tensor.expand(1, 3, -1, -1)
        total_dL_dB = total_dL_dB * valid_tensor.expand(1, 2, -1, -1)

        # Regularization (TV on local component, differentiable)
        loss_reg = compute_tv_regularization(
            G, B, self.config.lambda_G, self.config.lambda_B,
            TVMetricType.FROBENIUS)
        if torch.isnan(loss_reg):
            return None
        if loss_reg.requires_grad:
            loss_reg.backward(retain_graph=True)

        # Implicit differentiation: push eikonal gradients through CNN
        total_dL_dG = total_dL_dG.to(self.device)
        total_dL_dB = total_dL_dB.to(self.device)
        G2, B2, _, _ = self.model(spatial, global_f)
        G2.backward(total_dL_dG, retain_graph=True)
        B2.backward(total_dL_dB)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return {
            'loss': avg_data_loss + loss_reg.item(),
            'data_loss': avg_data_loss,
            'reg_loss': loss_reg.item(),
            'n_bands': n_loss_bands,
        }

    # -----------------------------------------------------------------
    # Band-based evaluation (validation / early stopping)
    # -----------------------------------------------------------------

    def evaluate_bands(self, scenarios):
        """Evaluate with non-overlapping bands and teacher forcing.

        Returns (aggregated_dict, per_fire_list).
        """
        self.model.eval()
        fire_metrics = []

        with torch.no_grad():
            for scenario in scenarios:
                spatial = scenario['spatial_feat'].unsqueeze(0).to(self.device)
                global_f = scenario['global_feat'].unsqueeze(0).to(self.device)
                G, B, _, _ = self.model(spatial, global_f)
                G_cpu, B_cpu = G.cpu(), B.cpu()

                true_T = torch.from_numpy(scenario['arrival_times']).float()
                valid = torch.from_numpy(scenario['valid_mask'])
                finite_valid = valid & torch.isfinite(true_T)
                if finite_valid.sum() < 10:
                    continue
                max_time = true_T[finite_valid].max().item()
                if max_time < 1e-6:
                    continue

                bands = self.generate_bands(max_time, overlapping=False)
                source_masks_K, valid_indices = self._build_band_sources(
                    scenario, bands, true_T, valid)
                if source_masks_K is None:
                    continue
                bands_valid = [bands[k] for k in valid_indices]

                T_batch = self._solve_bands_forward(
                    G_cpu, B_cpu, source_masks_K)
                if torch.isnan(T_batch).any():
                    continue

                band_ious, band_mses = [], []
                for k_idx, (t_start, t_end) in enumerate(bands_valid):
                    delta_t = t_end - t_start
                    if delta_t < 1e-6:
                        continue
                    source_2d = source_masks_K[k_idx]
                    not_source = valid & ~source_2d
                    gt_band = valid & (true_T > t_start) & (true_T <= t_end)

                    pred_burned = (T_batch[k_idx] <= delta_t) & not_source
                    inter = (pred_burned & gt_band).float().sum().item()
                    union = (pred_burned | gt_band).float().sum().item()
                    band_ious.append(inter / union if union > 0 else 0.0)

                    bp = gt_band & (T_batch[k_idx] < 1e5)
                    if bp.sum().item() > 0:
                        gt_rel = true_T[bp] - t_start
                        band_mses.append(
                            ((T_batch[k_idx][bp] - gt_rel) ** 2).mean().item())

                if band_ious:
                    fire_metrics.append({
                        'mean_band_iou': float(np.mean(band_ious)),
                        'band_ious': [float(x) for x in band_ious],
                        'mean_band_mse': (float(np.mean(band_mses))
                                          if band_mses else float('nan')),
                        'n_bands': len(band_ious),
                        'scene_id': scenario['scene_id'],
                        'event_id': scenario['event_id'],
                    })

        if not fire_metrics:
            return {'mean_band_iou_mean': 0.0, 'n_samples': 0}, fire_metrics

        ious = [m['mean_band_iou'] for m in fire_metrics]
        agg = {'mean_band_iou_mean': float(np.mean(ious)),
               'mean_band_iou_std': float(np.std(ious)),
               'n_samples': len(fire_metrics)}
        mses = [m['mean_band_mse'] for m in fire_metrics
                if not np.isnan(m['mean_band_mse'])]
        if mses:
            agg['mean_band_mse_mean'] = float(np.mean(mses))
            agg['mean_band_mse_std'] = float(np.std(mses))
        return agg, fire_metrics

    # -----------------------------------------------------------------
    # Full-fire evaluation (comparison with baseline)
    # -----------------------------------------------------------------

    def evaluate_full_fire(self, scenarios, save_predictions=False):
        """Solve from ignition, compute correlation / RMSE / IoU."""
        self.model.eval()
        metrics_list = []
        preds = {'pred_T': [], 'true_T': [], 'valid_mask': [], 'event_ids': []}

        with torch.no_grad():
            for s in scenarios:
                spatial = s['spatial_feat'].unsqueeze(0).to(self.device)
                global_f = s['global_feat'].unsqueeze(0).to(self.device)
                G, B, _, _ = self.model(spatial, global_f)
                source = torch.from_numpy(s['source_mask']).unsqueeze(0)
                prob = EikonalProblem(G.cpu(), B.cpu(), source)

                if HAVE_CUDA_SINGLE:
                    T_pred, _ = _cuda_single_solver(
                        prob, max_iter=self.config.max_iter)
                else:
                    T_pred, _ = fast_sweeping_solve_numba(
                        prob, max_iter=self.config.max_iter)

                m = compute_all_metrics(
                    T_pred[0].numpy(), s['arrival_times'], s['valid_mask'])
                m['scene_id'] = s['scene_id']
                m['event_id'] = s['event_id']
                metrics_list.append(m)

                if save_predictions:
                    preds['pred_T'].append(T_pred[0].numpy())
                    preds['true_T'].append(s['arrival_times'])
                    preds['valid_mask'].append(s['valid_mask'])
                    preds['event_ids'].append(s['event_id'])

        predictions = None
        if save_predictions:
            predictions = {k: np.array(v) for k, v in preds.items()}
        return aggregate_metrics(metrics_list), metrics_list, predictions

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------

    def train(self, train_scenarios, val_scenarios,
              verbose=True, scene_prefix=""):
        history = {
            'train_loss': [], 'train_data_loss': [], 'train_reg_loss': [],
            'val_band_iou': [], 'val_band_mse': [],
            'best_epoch': 0, 'best_val_iou': 0, 'learning_rate': [],
        }
        best_val_iou = -1
        best_state = None
        patience_counter = 0
        nan_streak = 0
        recovery_count = 0

        for epoch in range(self.config.n_epochs):
            t_epoch = time.time()
            losses, dloss, rloss, nbands = [], [], [], []
            n_skip = 0
            np.random.shuffle(train_scenarios)

            for fi, scenario in enumerate(train_scenarios):
                t_fire = time.time()
                r = self.train_step(scenario)
                dt = time.time() - t_fire

                if r is not None:
                    losses.append(r['loss'])
                    dloss.append(r['data_loss'])
                    rloss.append(r['reg_loss'])
                    nbands.append(r['n_bands'])
                    if verbose and (fi + 1) % 50 == 0:
                        print(f"{scene_prefix}  ep{epoch+1} fire {fi+1}/"
                              f"{len(train_scenarios)}: "
                              f"loss={r['loss']:.3f} bands={r['n_bands']} "
                              f"({dt:.1f}s)")
                else:
                    n_skip += 1

            if not losses:
                nan_streak += 1
                if verbose:
                    print(f"{scene_prefix}  Epoch {epoch+1}: all fires "
                          f"skipped (nan_streak={nan_streak})")
                if nan_streak >= 5 and best_state is not None:
                    recovery_count += 1
                    if recovery_count > 10:
                        if verbose:
                            print(f"{scene_prefix}  Too many recoveries")
                        break
                    if verbose:
                        print(f"{scene_prefix}  Recovery #{recovery_count}")
                    self.model.load_state_dict(best_state)
                    lr = self.config.learning_rate * (0.5 ** recovery_count)
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr=lr)
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode='max', factor=0.5,
                        patience=self.config.scheduler_patience, min_lr=1e-6)
                    nan_streak = 0
                continue

            nan_streak = 0
            al = np.mean(losses)
            ad = np.mean(dloss)
            ar = np.mean(rloss)
            ab = np.mean(nbands)
            history['train_loss'].append(al)
            history['train_data_loss'].append(ad)
            history['train_reg_loss'].append(ar)
            history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr'])

            dt_epoch = time.time() - t_epoch

            if (epoch + 1) % 5 == 0 or epoch == 0:
                va, _ = self.evaluate_bands(val_scenarios)
                vi = va.get('mean_band_iou_mean', 0)
                vm = va.get('mean_band_mse_mean', float('inf'))
                history['val_band_iou'].append(vi)
                history['val_band_mse'].append(vm)
                self.scheduler.step(vi)

                if vi > best_val_iou:
                    best_val_iou = vi
                    best_state = {k: v.clone()
                                  for k, v in self.model.state_dict().items()}
                    history['best_epoch'] = epoch + 1
                    history['best_val_iou'] = best_val_iou
                    patience_counter = 0
                else:
                    patience_counter += 5

                if verbose:
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"{scene_prefix}  Epoch {epoch+1:3d}: "
                          f"loss={al:.3f} (data={ad:.3f}, reg={ar:.4f}, "
                          f"bands={ab:.1f}), val_iou={vi:.3f}, "
                          f"val_mse={vm:.1f}, lr={lr:.1e}, "
                          f"skip={n_skip}, {dt_epoch:.0f}s")

                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"{scene_prefix}  Early stopping at "
                              f"epoch {epoch+1}")
                    break
            else:
                if verbose:
                    print(f"{scene_prefix}  Epoch {epoch+1:3d}: "
                          f"loss={al:.3f} (data={ad:.3f}, reg={ar:.4f}, "
                          f"bands={ab:.1f}), skip={n_skip}, {dt_epoch:.0f}s")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history
