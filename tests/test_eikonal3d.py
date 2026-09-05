"""CPU checks for the diagonal 3D reference, independent of wildfire code.

Run from the repository root with ``python -m unittest tests.test_eikonal3d``.
Arrays and sources are ZYX; speeds and physical spacings are XYZ (mm/ms, mm).
"""

import importlib.util
import unittest

import numpy as np
import torch

from eikonal.three_d.analytic import analytical_arrival
from eikonal.three_d.autograd import SharedSpeedModel, solve_eikonal
from eikonal.three_d.solver import local_update, solve_diagonal


def error_metrics(numerical, exact):
    error = np.asarray(numerical) - np.asarray(exact)
    return {
        "mae_ms": float(np.mean(np.abs(error))),
        "rmse_ms": float(np.sqrt(np.mean(error**2))),
        "max_abs_ms": float(np.max(np.abs(error))),
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(exact)),
    }


class AnalyticalReferenceTests(unittest.TestCase):
    def test_physical_axes_and_arbitrary_source(self):
        shape, source = (5, 7, 9), (1, 4, 2)
        speeds, spacing = (0.6, 0.4, 0.2), (0.7, 0.9, 1.1)
        arrival = analytical_arrival(shape, source, speeds, spacing)
        self.assertEqual(arrival.shape, shape)
        self.assertEqual(arrival.dtype, np.float64)
        self.assertEqual(arrival[source], 0.0)
        self.assertAlmostEqual(arrival[1, 4, 5], 3 * 0.7 / 0.6)
        self.assertAlmostEqual(arrival[1, 6, 2], 2 * 0.9 / 0.4)
        self.assertAlmostEqual(arrival[4, 4, 2], 3 * 1.1 / 0.2)
        expected = np.sqrt((3 * 0.7 / 0.6) ** 2 + (2 * 0.9 / 0.4) ** 2 + (3 * 1.1 / 0.2) ** 2)
        self.assertAlmostEqual(arrival[4, 6, 5], expected)

    def test_scalar_isotropy_and_explicit_diagonal_agree(self):
        scalar = analytical_arrival((5, 6, 7), (0, 2, 6), 0.51)
        diagonal = analytical_arrival((5, 6, 7), (0, 2, 6), (0.51,) * 3)
        np.testing.assert_array_equal(scalar, diagonal)


class LocalUpdateTests(unittest.TestCase):
    def check_equation(self, times, weights):
        times, weights = np.asarray(times, dtype=float), np.asarray(weights, dtype=float)
        root, active = local_update(times, weights)
        active = np.asarray(active, dtype=bool)
        self.assertTrue(np.isfinite(root))
        np.testing.assert_array_equal(active, np.isfinite(times) & (times < root))
        delta = root - times[active]
        self.assertTrue(np.all(delta > 0))
        self.assertAlmostEqual(float(np.dot(weights[active], delta**2)), 1.0, places=11)
        return root, active

    def test_edge_face_and_full_volume_branches(self):
        root, active = self.check_equation([0.0, 1.2, 10.0], [1.0] * 3)
        self.assertEqual(root, 1.0)
        np.testing.assert_array_equal(active, [True, False, False])
        root, active = self.check_equation([0.0, 0.2, 3.0], [1.0] * 3)
        self.assertAlmostEqual(root, 0.8)
        np.testing.assert_array_equal(active, [True, True, False])
        root, active = self.check_equation([0.0] * 3, [1.0] * 3)
        self.assertAlmostEqual(root, 1 / np.sqrt(3))
        self.assertTrue(active.all())

    def test_zero_contribution_activation_tie_is_inactive(self):
        root, active = self.check_equation([0.0, 1.0, np.inf], [1.0] * 3)
        self.assertEqual(root, 1.0)
        np.testing.assert_array_equal(active, [True, False, False])

    def test_donor_activation_switch_has_matching_first_derivative(self):
        # At t=u_y=1 the y contribution is zero. The squared positive-part
        # residual is C1 here, even though the retained active set changes.
        times = np.array([0.0, 1.0, np.inf])
        step = 1e-6
        upper, active_upper = self.check_equation(times, [1.0 + step, 1.0, 1.0])
        lower, active_lower = self.check_equation(times, [1.0 - step, 1.0, 1.0])
        self.assertFalse(active_upper[1])
        self.assertTrue(active_lower[1])
        derivative_wx = (upper - lower) / (2 * step)
        self.assertAlmostEqual(derivative_wx, -0.5, delta=2e-6)

    def test_randomized_local_residual_and_monotonicity(self):
        rng = np.random.default_rng(2905)
        for _ in range(100):
            times = rng.uniform(0, 5, size=3)
            weights = np.exp(rng.uniform(-2, 2, size=3))
            root, _ = self.check_equation(times, weights)
            raised, _ = local_update(times + rng.uniform(0, 0.2, size=3), weights)
            self.assertGreaterEqual(raised + 1e-12, root)

    def test_time_translation_does_not_destabilize_quadratic(self):
        times = np.array([0.0, 0.2, 0.5])
        weights = np.array([0.8, 1.1, 0.5])
        root, active = local_update(times, weights)
        translated, translated_active = local_update(times + 1e8, weights)
        self.assertLess(abs((translated - 1e8) - root), 3e-8)
        np.testing.assert_array_equal(active, translated_active)


class ForwardSolverTests(unittest.TestCase):
    def check_solution(self, shape, source, speeds, spacing=(0.8, 0.8, 0.8), **kwargs):
        result = solve_diagonal(shape, source, speeds, spacing, **kwargs)
        self.assertEqual(result.arrival.shape, shape)
        self.assertTrue(np.isfinite(result.arrival).all())
        self.assertEqual(result.arrival[source], 0.0)
        self.assertGreaterEqual(result.cycles, 1)
        self.assertLessEqual(result.max_residual, 1e-8)
        self.assertLessEqual(result.max_fixed_point_error, 1e-8)
        flat = result.arrival.ravel()
        donors = result.donors
        self.assertEqual(donors.shape, (flat.size, 3))
        for axis in range(3):
            chosen = donors[:, axis] >= 0
            self.assertTrue(np.all(flat[donors[chosen, axis]] < flat[chosen]))
        return result

    def test_sphere_central_off_center_and_boundary_sources(self):
        shape = (9, 11, 13)
        for source in [(4, 5, 6), (2, 7, 3), (0, 0, 0), (8, 10, 12)]:
            with self.subTest(source=source):
                result = self.check_solution(shape, source, (0.5,) * 3)
                exact = analytical_arrival(shape, source, 0.5)
                metrics = error_metrics(result.arrival, exact)
                self.assertLess(metrics["relative_l2"], 0.22)
                self.assertGreaterEqual(float(np.min(result.arrival - exact)), -1e-10)

    def test_ellipsoid_physical_axis_rays_and_nonuniform_spacing(self):
        shape, source = (9, 11, 13), (3, 5, 4)
        speeds, spacing = (0.61, 0.39, 0.27), (0.7, 0.9, 1.1)
        result = self.check_solution(shape, source, speeds, spacing)
        exact = analytical_arrival(shape, source, speeds, spacing)
        np.testing.assert_allclose(result.arrival[source[0], source[1], :], exact[source[0], source[1], :], atol=1e-10)
        np.testing.assert_allclose(result.arrival[source[0], :, source[2]], exact[source[0], :, source[2]], atol=1e-10)
        np.testing.assert_allclose(result.arrival[:, source[1], source[2]], exact[:, source[1], source[2]], atol=1e-10)
        self.assertLess(error_metrics(result.arrival, exact)["relative_l2"], 0.22)

    def test_all_nine_source_depths_propagate_between_layers(self):
        shape, speeds = (9, 7, 9), (0.58, 0.41, 0.29)
        for z in range(shape[0]):
            source = (z, 3, 4)
            result = self.check_solution(shape, source, speeds)
            expected_depth = np.abs(np.arange(shape[0]) - z) * 0.8 / speeds[2]
            np.testing.assert_allclose(result.arrival[:, 3, 4], expected_depth, atol=1e-10)
            self.assertEqual(np.count_nonzero(result.arrival == 0), 1)

    def test_refinement_reduces_sphere_and_ellipsoid_error(self):
        for name, speeds, source12 in [
            ("sphere_center", (0.5,) * 3, (6, 6, 6)),
            ("sphere_boundary", (0.5,) * 3, (0, 4, 9)),
            ("ellipsoid", (0.61, 0.39, 0.27), (3, 4, 7)),
        ]:
            errors = []
            for resolution in (13, 25):
                scale = (resolution - 1) // 12
                shape = (resolution,) * 3
                source = tuple(value * scale for value in source12)
                spacing = (0.8 / scale,) * 3
                result = self.check_solution(shape, source, speeds, spacing)
                exact = analytical_arrival(shape, source, speeds, spacing)
                metrics = error_metrics(result.arrival, exact)
                errors.append(metrics["mae_ms"])
                print("FORWARD {} n={} h_mm={:.3g} MAE_ms={:.8f} RMSE_ms={:.8f} max_ms={:.8f} relL2={:.8f} cycles={} residual={:.2e}".format(
                    name, resolution, spacing[0], metrics["mae_ms"], metrics["rmse_ms"], metrics["max_abs_ms"], metrics["relative_l2"], result.cycles, result.max_residual
                ))
            self.assertLess(errors[1], 0.8 * errors[0], name)

    def test_cardiac_slab_all_outer_planes_are_finite(self):
        result = self.check_solution((9, 49, 49), (4, 24, 24), (0.5754, 0.42416, 0.27059))
        self.assertTrue(np.isfinite(result.arrival[[0, -1], :, :]).all())
        self.assertTrue(np.isfinite(result.arrival[:, [0, -1], :]).all())
        self.assertTrue(np.isfinite(result.arrival[:, :, [0, -1]]).all())

    def test_axis_permutation_and_reflection_equivariance(self):
        shape, source = (5, 7, 9), (1, 4, 2)
        speeds, spacing = (0.57, 0.39, 0.28), (0.7, 0.9, 1.1)
        original = self.check_solution(shape, source, speeds, spacing).arrival
        # Reverse XYZ and ZYX consistently by exchanging the x and z axes.
        permuted = self.check_solution(shape[::-1], source[::-1], speeds[::-1], spacing[::-1]).arrival
        np.testing.assert_allclose(permuted, original.transpose(2, 1, 0), atol=1e-10)
        reflected_source = tuple(n - 1 - s for n, s in zip(shape, source))
        reflected = self.check_solution(shape, reflected_source, speeds, spacing).arrival
        np.testing.assert_allclose(reflected, original[::-1, ::-1, ::-1], atol=1e-10)

    def test_one_voxel_axes_have_lower_dimensional_fallbacks(self):
        for shape, source in [((1, 1, 7), (0, 0, 2)), ((1, 5, 7), (0, 1, 2)), ((5, 1, 7), (1, 0, 2))]:
            self.check_solution(shape, source, (0.6, 0.4, 0.2))

    def test_insufficient_sweeps_raise(self):
        with self.assertRaises(RuntimeError):
            solve_diagonal((7, 8, 9), (2, 3, 4), (0.6, 0.4, 0.3), max_cycles=1)

    def test_invalid_speed_and_spacing_are_rejected(self):
        for speeds in [(0.0, 0.4, 0.3), (-0.6, 0.4, 0.3), (np.nan, 0.4, 0.3), (np.inf, 0.4, 0.3)]:
            with self.assertRaises(ValueError):
                solve_diagonal((3, 4, 5), (1, 2, 3), speeds)
        with self.assertRaises(ValueError):
            solve_diagonal((3, 4, 5), (1, 2, 3), (0.6, 0.4, 0.3), spacing_xyz=(0.0, 0.8, 0.8))

    @unittest.skipUnless(importlib.util.find_spec("numba"), "optional numba is unavailable")
    def test_python_and_numba_execute_the_same_update(self):
        args = ((5, 7, 9), (1, 4, 2), (0.57, 0.39, 0.28), (0.7, 0.9, 1.1))
        pure = self.check_solution(*args, backend="python")
        compiled = self.check_solution(*args, backend="numba")
        np.testing.assert_allclose(pure.arrival, compiled.arrival, atol=1e-12, rtol=1e-12)
        np.testing.assert_array_equal(pure.donors, compiled.donors)


class SharedParameterTests(unittest.TestCase):
    def test_discrete_scaling_and_implicit_gradient_identity(self):
        shape, source = (5, 7, 9), (1, 4, 2)
        speeds = torch.tensor([0.57, 0.39, 0.28], dtype=torch.float64, requires_grad=True)
        arrival = solve_eikonal(speeds, shape, source)
        weights = torch.linspace(-0.3, 1.7, arrival.numel(), dtype=arrival.dtype).reshape(shape)
        loss = (arrival * weights).mean()
        gradient, = torch.autograd.grad(loss, speeds)
        self.assertAlmostEqual(float(torch.dot(speeds.detach(), gradient)), -float(loss.detach()), places=9)
        scaled = solve_eikonal(speeds.detach() * 1.37, shape, source)
        torch.testing.assert_close(scaled, arrival.detach() / 1.37, atol=1e-10, rtol=1e-11)

    def test_source_has_zero_speed_gradient(self):
        speeds = torch.tensor([0.57, 0.39, 0.28], dtype=torch.float64, requires_grad=True)
        source = (1, 4, 2)
        arrival = solve_eikonal(speeds, (5, 7, 9), source)
        arrival[source].backward()
        torch.testing.assert_close(speeds.grad, torch.zeros_like(speeds))

    def test_separate_source_solutions_accumulate_into_shared_speeds(self):
        speeds = torch.tensor([0.57, 0.39, 0.28], dtype=torch.float64, requires_grad=True)
        shape, sources = (5, 7, 9), [(1, 4, 2), (4, 0, 8)]
        losses = [solve_eikonal(speeds, shape, source).mean() for source in sources]
        individual = [torch.autograd.grad(loss, speeds, retain_graph=True)[0] for loss in losses]
        shared, = torch.autograd.grad(sum(losses), speeds)
        torch.testing.assert_close(shared, individual[0] + individual[1], atol=1e-11, rtol=1e-11)
        self.assertTrue(torch.isfinite(shared).all())
        self.assertTrue((shared < 0).all())

    def test_both_shared_models_take_a_finite_optimizer_step(self):
        for isotropic, initial in [(True, 0.5), (False, (0.6, 0.4, 0.3))]:
            model = SharedSpeedModel(initial_speeds=initial, isotropic=isotropic)
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
            before = model.speeds_xyz.detach().clone()
            optimizer.zero_grad()
            arrival = model((5, 7, 9), (1, 4, 2))
            loss = arrival.mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
            optimizer.step()
            after = model.speeds_xyz.detach()
            self.assertTrue(torch.isfinite(after).all())
            self.assertTrue((after > before).all())
            self.assertTrue((after > 0).all())
            if isotropic:
                self.assertEqual(float(after[0]), float(after[1]))
                self.assertEqual(float(after[1]), float(after[2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
