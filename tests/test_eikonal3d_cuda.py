"""Tiny tensor-equation checks; --require-cuda also requires real GPU parity."""
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from eikonal.three_d import solve_diagonal, solve_eikonal, SharedSpeedModel
from eikonal.three_d.cuda import _TensorArrival, _forward


def parity(device):
    maximum_forward = maximum_gradient = 0.
    for shape, source, values in [((3, 4, 5), (1, 2, 3), (.57, .39, .28)),
                                  ((4, 5, 6), (0, 4, 0), (.4, .4, .4)),
                                  ((1, 4, 5), (0, 1, 3), (.2, .8, 1.1))]:
        spacing = (.8, .6, .9)
        cpu = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        reference = solve_eikonal(cpu, shape, source, spacing, backend="numba")
        parameters = cpu.detach().to(device).requires_grad_()
        output = (solve_eikonal(parameters, shape, source, spacing, backend="cuda")
                  if device == "cuda" else _TensorArrival.apply(parameters, shape, source, spacing))
        assert output.device.type == device
        weights = torch.linspace(-.4, 1.7, reference.numel(), dtype=torch.float64).reshape(shape)
        expected_gradient, = torch.autograd.grad((reference * weights).mean(), cpu)
        actual_gradient, = torch.autograd.grad((output * weights.to(device)).mean(), parameters)
        torch.testing.assert_close(output.cpu(), reference, atol=1e-10, rtol=1e-11)
        torch.testing.assert_close(actual_gradient.cpu(), expected_gradient, atol=1e-10, rtol=1e-10)
        maximum_forward = max(maximum_forward, float((output.cpu() - reference).abs().max().detach()))
        maximum_gradient = max(maximum_gradient, float((actual_gradient.cpu() - expected_gradient).abs().max()))
        if device == "cuda":
            wrapped = solve_diagonal(shape, source, values, spacing, backend="cuda")
            assert wrapped.backend == "cuda"
            np.testing.assert_allclose(wrapped.arrival, reference.detach().numpy(), atol=1e-10)
    print(f"{device}: forward max error={maximum_forward:.3g}; speed-gradient max error={maximum_gradient:.3g}")


class TensorEquationTests(unittest.TestCase):
    def test_reference_equations_and_shared_speed_gradients(self):
        parity("cpu")

    def test_scalar_parameter_gradient_and_source(self):
        speed = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
        output = _TensorArrival.apply(speed.expand(3), (3, 4, 5), (1, 2, 3), (.8, .8, .8))
        output.mean().backward()
        self.assertAlmostEqual(float(speed.detach() * speed.grad), -float(output.detach().mean()), places=10)
        self.assertEqual(float(output[1, 2, 3].detach()), 0.)

    def test_cpu_backend_and_cuda_selection_are_explicit(self):
        model = SharedSpeedModel(.5)
        with self.assertRaisesRegex(ValueError, "CUDA speed tensors"):
            model((2, 3, 4), (0, 1, 2), backend="cuda")

    def test_tensor_backward_decreases_tiny_fit_loss(self):
        shape, source, spacing = (3, 4, 5), (1, 2, 3), (.8, .8, .8)
        target = torch.from_numpy(solve_diagonal(shape, source, (.57, .39, .28), spacing,
                                                backend="numba").arrival)
        model = SharedSpeedModel((.57 * .85, .39 * .85, .28 * .85), isotropic=False)
        optimizer = torch.optim.LBFGS(model.parameters(), max_iter=3, line_search_fn="strong_wolfe")

        def objective():
            return (_TensorArrival.apply(model.speeds_xyz, shape, source, spacing) - target).square().mean()

        def closure():
            optimizer.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        before = float(objective().detach())
        optimizer.step(closure)
        self.assertLess(float(objective().detach()), before)
        self.assertTrue(torch.isfinite(model.log_speeds.grad).all())


@unittest.skipUnless(torch.cuda.is_available(), "Real CUDA device required")
class CudaTests(unittest.TestCase):
    def test_forward_gradient_parity(self):
        parity("cuda")

    def test_tiny_existing_cardiac_fit(self):
        from experiments.cardiac.numerical_training import fit_numerical
        shape, source, spacing = (3, 4, 5), (1, 2, 3), (.8, .8, .8)
        for isotropic, truth in [(True, [.55, .55, .55]), (False, [.57, .39, .28])]:
            target = solve_diagonal(shape, source, truth, spacing, backend="numba").arrival + 1.0529606913652159
            data = SimpleNamespace(shape_zyx=shape, spacing_xyz=spacing,
                                   sources_nzyx=np.array([source]), activation_nzyx=target[None])
            result = fit_numerical(data, [0], 1.0529606913652159, np.asarray(truth) * .85,
                                   isotropic, max_iter=3, backend="cuda")
            self.assertTrue(result["loss_decreased"])
            self.assertTrue(result["finite_gradients"])
        print("CUDA shared isotropic/XYZ cardiac optimization verified on", torch.cuda.get_device_name())


if __name__ == "__main__":
    if "--require-cuda" in sys.argv:
        sys.argv.remove("--require-cuda")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable: GPU parity/smoke was NOT run")
    unittest.main(verbosity=2)
