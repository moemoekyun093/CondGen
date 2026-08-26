import unittest
import math

import numpy as np
import torch

from tabdiff.models.doob_h_transform import (
    CategoricalHTransformGuide,
    NumericalBoxQuery,
    NumericalDoobHGuide,
    NumericalHScoreGuide,
    categorical_log_h_ratios,
)
from tabdiff.doob_h_runtime import infer_denoiser_type
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion


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

    def test_contains_ignores_inactive_columns(self):
        query = NumericalBoxQuery(
            lower=torch.tensor([-1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        rows = torch.tensor([[5.0, 1.0], [0.0, 3.0], [5.0, 3.0]])
        active = torch.tensor([0, 1])
        self.assertEqual(
            query.contains(rows, active).tolist(),
            [True, False, False],
        )
        self.assertEqual(
            query.contains(rows, torch.zeros(2)).tolist(),
            [True, True, True],
        )

    def test_round_trip(self):
        query = NumericalBoxQuery(
            lower=torch.tensor([-1.0, 0.0]),
            upper=torch.tensor([1.0, 2.0]),
        )
        restored = NumericalBoxQuery.from_dict(query.to_dict())
        torch.testing.assert_close(restored.lower, query.lower)
        torch.testing.assert_close(restored.upper, query.upper)


class NumericalDoobHGuideTest(unittest.TestCase):
    def test_separate_guides_have_independent_matching_backbones(self):
        kwargs = dict(
            d_numerical=3,
            categories=[2],
            d_token=16,
            num_layers=1,
            n_head=4,
            n_frequencies=4,
            query_mask_conditioning=True,
        )
        numerical = NumericalHScoreGuide(**kwargs)
        categorical = CategoricalHTransformGuide(**kwargs)
        x_num = torch.randn(4, 3)
        x_cat = torch.zeros(4, 2)
        x_cat[:, 0] = 1
        t = torch.linspace(0.1, 0.9, 4)
        active = torch.ones(4, 3)

        self.assertIsNone(numerical.h_logit_head)
        self.assertIsNone(categorical.correction_head)
        self.assertIsNot(numerical.tokenizer, categorical.tokenizer)
        self.assertEqual(numerical(x_num, x_cat, t, active).shape, (4, 3))
        self.assertEqual(categorical.log_h(x_num, x_cat, t, active).shape, (4,))

    def test_scalar_h_gradient_is_autograd_of_same_log_h(self):
        guide = NumericalDoobHGuide(
            d_numerical=3,
            categories=[2],
            d_token=16,
            num_layers=1,
            n_head=4,
            n_frequencies=4,
            query_mask_conditioning=True,
            scalar_h_gradient=True,
        )
        guide.eval()
        with torch.no_grad():
            guide.h_logit_head.weight.normal_(mean=0.0, std=0.1)
        x_num = torch.randn(4, 3)
        x_cat = torch.zeros(4, 2)
        x_cat[:, 0] = 1
        t = torch.linspace(0.1, 0.9, 4)
        active = torch.ones(4, 3)

        x_expected = x_num.detach().requires_grad_(True)
        expected = torch.autograd.grad(
            guide.log_h(x_expected, x_cat, t, active).sum(),
            x_expected,
        )[0]
        actual = guide(x_num, x_cat, t, active)

        self.assertIsNone(guide.correction_head)
        torch.testing.assert_close(actual, expected)

    def test_partial_query_mask_is_an_explicit_input(self):
        guide = NumericalDoobHGuide(
            d_numerical=3,
            d_token=16,
            num_layers=1,
            n_head=4,
            n_frequencies=4,
            query_mask_conditioning=True,
        )
        x_num = torch.randn(2, 3)
        t = torch.tensor([0.2, 0.7])
        full_tokens = guide._encode(x_num, None, t, torch.ones(2, 3))
        partial_tokens = guide._encode(
            x_num,
            None,
            t,
            torch.tensor([[1, 0, 1], [1, 0, 1]]),
        )
        self.assertFalse(torch.allclose(full_tokens, partial_tokens))
        self.assertTrue(guide.config_dict()["query_mask_conditioning"])

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
        torch.testing.assert_close(
            guide.log_h(x_num, x_cat, torch.linspace(0, 1, 5)),
            torch.full((5,), -math.log(2.0)),
        )

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


class CategoricalDoobUpdateTest(unittest.TestCase):
    def test_shared_ratio_helper_is_differentiable(self):
        class LinearLogH(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0, -1.0, 0.0]))

            def log_h(self, x_num, x_cat, t, query_active_mask=None):
                return (x_cat * self.weight).sum(dim=1)

        guide = LinearLogH()
        x_cat = torch.tensor([[2], [2]])

        def to_one_hot(values):
            return torch.nn.functional.one_hot(values[:, 0], num_classes=3).float()

        ratios = categorical_log_h_ratios(
            guide,
            x_num_t=torch.zeros(2, 1),
            x_cat_t=x_cat,
            t=torch.tensor([0.2, 0.8]),
            num_classes=[2],
            mask_index=torch.tensor([2]),
            to_one_hot=to_one_hot,
        )
        ratios[:, :, :2].sum().backward()

        self.assertEqual(ratios.shape, (2, 1, 3))
        self.assertIsNotNone(guide.weight.grad)

    def make_diffusion(self, num_classes=np.array([2])):
        return UnifiedCtimeDiffusion(
            num_classes=num_classes,
            num_numerical_features=1,
            denoise_fn=torch.nn.Identity(),
            y_only_model=None,
            noise_schedule_params={},
            sampler_params={},
            device=torch.device("cpu"),
        )

    def test_h_ratio_reweights_existing_mdlm_transition(self):
        diffusion = self.make_diffusion()
        log_p_x0 = torch.log(torch.tensor([[[0.5, 0.5, 1e-30]]]))
        x = torch.tensor([[2]])
        alpha_t = torch.tensor([[0.5]])
        alpha_s = torch.tensor([[0.8]])
        h_log_ratios = torch.zeros((1, 1, 3))
        h_log_ratios[0, 0, 0] = torch.log(torch.tensor(2.0))

        _, transition_weights = diffusion._mdlm_update(
            log_p_x0,
            x,
            alpha_t,
            alpha_s,
            h_log_ratios=h_log_ratios,
        )

        expected = torch.tensor([[[0.3, 0.15, 0.2]]])
        torch.testing.assert_close(transition_weights, expected)

    def test_scalar_h_gradient_is_scaled_by_sigma_squared(self):
        diffusion = self.make_diffusion()

        class ConstantGradientGuide(torch.nn.Module):
            scalar_h_gradient = True

            def forward(self, x_num_t, x_cat_t, t, query_active_mask=None):
                return torch.ones_like(x_num_t)

        diffusion.set_numerical_h_guide(ConstantGradientGuide())
        corrected = diffusion._apply_numerical_h_guide(
            denoised=torch.zeros(2, 1),
            x_num_t=torch.zeros(2, 1),
            x_cat_t=torch.zeros(2, 0),
            t=torch.tensor([0.2, 0.8]),
            sigma=torch.tensor([[2.0], [3.0]]),
        )
        torch.testing.assert_close(corrected, torch.tensor([[4.0], [9.0]]))

    def test_section4_posterior_matches_ordering_density(self):
        diffusion = self.make_diffusion(np.array([2, 3]))
        t = torch.linspace(0, 1, 1001)[:, None]
        weights = diffusion._section4_start_time_weights([0], t)

        sigma = diffusion.cat_schedule.total_noise(t)
        rate = diffusion.cat_schedule.rate_noise(t)
        alpha = torch.exp(-sigma)
        expected = alpha[:, 0] * rate[:, 0] * (1.0 - alpha[:, 0])
        expected = expected / expected.sum()

        self.assertAlmostEqual(weights.sum().item(), 1.0, places=6)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())
        torch.testing.assert_close(weights, expected)

    def test_section4_posterior_requires_fixed_column(self):
        diffusion = self.make_diffusion(np.array([2, 3]))
        with self.assertRaisesRegex(ValueError, "at least one fixed"):
            diffusion._section4_start_time_weights(
                [], torch.linspace(0, 1, 11)[:, None]
            )


if __name__ == "__main__":
    unittest.main()
