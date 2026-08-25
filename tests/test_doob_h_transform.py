import unittest

import torch

from tabdiff.models.doob_h_transform import NumericalBoxQuery, NumericalDoobHGuide
from tabdiff.doob_h_runtime import infer_denoiser_type


class CheckpointArchitectureTest(unittest.TestCase):
    def test_infers_original_architecture(self):
        state = {"denoise_fn_D.denoise_fn_F.encoder.layers.0.weight": object()}
        self.assertEqual(infer_denoiser_type(state), "original")

    def test_infers_ft_periodic_architecture(self):
        state = {"denoise_fn_D.denoise_fn_F.blocks.0.norm1.weight": object()}
        self.assertEqual(infer_denoiser_type(state), "ft_periodic")

    def test_infers_tabnet_architecture(self):
        state = {"denoise_fn_D.denoise_fn_F.tabnet_steps.0.weight": object()}
        self.assertEqual(infer_denoiser_type(state), "tabnet")


class NumericalBoxQueryTest(unittest.TestCase):
    def test_contains_requires_every_numeric_column(self):
        query = NumericalBoxQuery(
            lower=torch.tensor([-1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        rows = torch.tensor([[0.0, 1.0], [2.0, 1.0], [0.0, 3.0]])
        self.assertEqual(query.contains(rows).tolist(), [True, False, False])

    def test_round_trip(self):
        query = NumericalBoxQuery(
            lower=torch.tensor([-1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        restored = NumericalBoxQuery.from_dict(query.to_dict())
        torch.testing.assert_close(restored.lower, query.lower)
        torch.testing.assert_close(restored.upper, query.upper)


class NumericalDoobHGuideTest(unittest.TestCase):
    def test_mixed_context_produces_numeric_correction_only(self):
        guide = NumericalDoobHGuide(
            d_numerical=3,
            categories=[3, 2],
            d_token=16,
            num_layers=1,
            n_head=4,
            n_frequencies=4,
        )
        x_num = torch.randn(5, 3)
        x_cat = torch.zeros(5, 5)
        x_cat[:, 0] = 1
        x_cat[:, 3] = 1
        output = guide(x_num, x_cat, torch.linspace(0, 1, 5))
        self.assertEqual(output.shape, (5, 3))
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_numerical_only_forward_and_config_round_trip(self):
        guide = NumericalDoobHGuide(
            d_numerical=4,
            d_token=16,
            num_layers=1,
            n_head=4,
            n_frequencies=4,
        )
        restored = NumericalDoobHGuide(**guide.config_dict())
        output = restored(torch.randn(2, 4), None, torch.tensor(0.5))
        self.assertEqual(output.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
