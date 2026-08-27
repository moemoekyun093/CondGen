import unittest
import math

import numpy as np
import torch

from tabdiff.models.doob_h_transform import (
    CategoricalHTransformGuide,
    NumericalBoxQuery,
    NumericalDoobHGuide,
    NumericalHScoreGuide,
    categorical_candidate_log_h,
    eligible_row_indices,
    guided_categorical_log_probs,
    sample_conditional_batch,
    sample_constraint_mask,
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


class PartialConstraintBatchTest(unittest.TestCase):
    def test_eligible_rows_follow_only_active_constraints(self):
        row_satisfies = torch.tensor(
            [
                [True, True, False],
                [True, False, True],
                [False, True, True],
                [True, True, True],
            ]
        )
        active = torch.tensor([1.0, 0.0, 1.0])
        self.assertEqual(
            eligible_row_indices(row_satisfies, active).tolist(),
            [1, 3],
        )
        self.assertEqual(
            eligible_row_indices(row_satisfies, torch.zeros(3)).tolist(),
            [0, 1, 2, 3],
        )

    def test_anchor_masks_are_exact(self):
        all_active, active_kind = sample_constraint_mask(
            4, torch.device("cpu"), torch.float32, 0.5, 1.0, 0.0
        )
        all_inactive, inactive_kind = sample_constraint_mask(
            4, torch.device("cpu"), torch.float32, 0.5, 0.0, 1.0
        )
        torch.testing.assert_close(all_active, torch.ones(4))
        torch.testing.assert_close(all_inactive, torch.zeros(4))
        self.assertEqual(active_kind, "all_active")
        self.assertEqual(inactive_kind, "all_inactive")

    def test_conditional_batch_uses_only_eligible_rows(self):
        rows = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        eligible = torch.tensor([1, 4])
        batch = sample_conditional_batch(rows, eligible, batch_size=100)
        allowed_first_values = {rows[1, 0].item(), rows[4, 0].item()}
        self.assertTrue(set(batch[:, 0].tolist()).issubset(allowed_first_values))


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
            query_mask_fusion="concat",
            query_mask_embedding_dim=4,
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
        self.assertIsNot(
            numerical.query_mask_embedding,
            categorical.query_mask_embedding,
        )
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
            query_mask_fusion="concat",
            query_mask_embedding_dim=4,
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
        self.assertEqual(guide.config_dict()["query_mask_fusion"], "concat")
        self.assertEqual(guide.query_mask_embedding.weight.shape, (2, 4))
        self.assertEqual(guide.query_token_fusion[0].in_features, 20)
        self.assertIsNone(guide.query_active_embed)

    def test_legacy_additive_mask_architecture_still_loads(self):
        kwargs = dict(
            d_numerical=3,
            d_token=16,
            num_layers=1,
            n_head=4,
            n_frequencies=4,
            query_mask_conditioning=True,
        )
        original = NumericalDoobHGuide(**kwargs)
        restored = NumericalDoobHGuide(**kwargs)
        restored.load_state_dict(original.state_dict())

        self.assertEqual(restored.config_dict()["query_mask_fusion"], "additive")
        self.assertIsNotNone(restored.query_active_embed)
        self.assertIsNone(restored.query_mask_embedding)

    def test_concat_fusion_rejects_nonbinary_mask(self):
        guide = NumericalDoobHGuide(
            d_numerical=2,
            d_token=16,
            num_layers=1,
            n_head=4,
            n_frequencies=4,
            query_mask_conditioning=True,
            query_mask_fusion="concat",
        )
        with self.assertRaisesRegex(ValueError, "must be binary"):
            guide._encode(
                torch.randn(1, 2),
                None,
                torch.tensor([0.5]),
                torch.tensor([[1.0, 0.5]]),
            )

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

        scores = categorical_candidate_log_h(
            guide,
            x_num_t=torch.zeros(2, 1),
            x_cat_t=x_cat,
            t=torch.tensor([0.2, 0.8]),
            num_classes=[2],
            mask_index=torch.tensor([2]),
            to_one_hot=to_one_hot,
        )
        scores[:, :, :2].sum().backward()

        self.assertEqual(scores.shape, (2, 1, 3))
        self.assertIsNotNone(guide.weight.grad)

    def test_candidate_helper_skips_current_state_and_uses_whole_row(self):
        class WholeRowLogH(torch.nn.Module):
            def log_h(self, x_num, x_cat, t, query_active_mask=None):
                # With one categorical column, index 2 is MASK.  Candidate-only
                # evaluation must never pass the unchanged masked state here.
                if torch.any(x_cat[:, 2] != 0):
                    raise AssertionError("current masked state was evaluated")
                return x_num[:, 0] + 3.0 * x_cat[:, 1] + t

        def to_one_hot(values):
            return torch.nn.functional.one_hot(values[:, 0], num_classes=3).float()

        scores = categorical_candidate_log_h(
            WholeRowLogH(),
            x_num_t=torch.tensor([[2.0]]),
            x_cat_t=torch.tensor([[2]]),
            t=torch.tensor([0.5]),
            num_classes=[2],
            mask_index=torch.tensor([2]),
            to_one_hot=to_one_hot,
        )

        torch.testing.assert_close(scores[0, 0, :2], torch.tensor([2.5, 5.5]))

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

    def test_candidate_log_h_changes_category_not_reveal_probability(self):
        diffusion = self.make_diffusion()
        log_p_x0 = torch.log(torch.tensor([[[0.5, 0.5, 1e-30]]]))
        x = torch.tensor([[2]])
        alpha_t = torch.tensor([[0.5]])
        alpha_s = torch.tensor([[0.8]])
        h_candidate_log_scores = torch.zeros((1, 1, 3))
        h_candidate_log_scores[0, 0, 0] = torch.log(torch.tensor(2.0))

        _, transition_weights = diffusion._mdlm_update(
            log_p_x0,
            x,
            alpha_t,
            alpha_s,
            h_candidate_log_scores=h_candidate_log_scores,
        )

        # The original real-token mass is (1-alpha_t)-(1-alpha_s)=0.3
        # and MASK mass is 1-alpha_s=0.2.  h changes only the 2:1 split.
        expected = torch.tensor([[[0.2, 0.1, 0.2]]])
        torch.testing.assert_close(transition_weights, expected)
        self.assertAlmostEqual(
            transition_weights[0, 0, :2].sum().item(),
            0.3,
            places=6,
        )
        self.assertAlmostEqual(transition_weights[0, 0, 2].item(), 0.2, places=6)

    def test_constant_h_recovers_original_transition(self):
        diffusion = self.make_diffusion()
        log_p_x0 = torch.log(torch.tensor([[[0.25, 0.75, 1e-30]]]))
        x = torch.tensor([[2]])
        alpha_t = torch.tensor([[0.5]])
        alpha_s = torch.tensor([[0.8]])

        _, base_weights = diffusion._mdlm_update(
            log_p_x0,
            x,
            alpha_t,
            alpha_s,
        )
        _, guided_weights = diffusion._mdlm_update(
            log_p_x0,
            x,
            alpha_t,
            alpha_s,
            h_candidate_log_scores=torch.full_like(log_p_x0, 3.0),
        )
        torch.testing.assert_close(guided_weights, base_weights)

    def test_guided_endpoint_law_is_normalized(self):
        base = torch.log(torch.tensor([[[0.2, 0.8, 1e-30]]]))
        candidate_scores = torch.tensor([[[math.log(4.0), 0.0, 0.0]]])
        guided = guided_categorical_log_probs(base, candidate_scores).exp()

        torch.testing.assert_close(guided.sum(dim=-1), torch.ones((1, 1)))
        torch.testing.assert_close(guided[0, 0, :2], torch.tensor([0.5, 0.5]))

    def test_fixed_rate_generator_matching_equals_weighted_cross_entropy(self):
        base = torch.log(torch.tensor([[[0.25, 0.75, 1e-30]]]))
        candidate_scores = torch.tensor([[[0.3, -0.2, 0.0]]])
        guided = guided_categorical_log_probs(base, candidate_scores).exp()
        weight = torch.tensor(2.0)
        endpoint = 1

        predicted_rates = weight * guided[0, 0, :2]
        target_rates = torch.tensor([0.0, weight.item()])
        positive = target_rates > 0
        rate_kl = (
            predicted_rates.sum()
            - target_rates.sum()
            + (
                target_rates[positive]
                * torch.log(
                    target_rates[positive] / predicted_rates[positive]
                )
            ).sum()
        )
        weighted_cross_entropy = -weight * torch.log(guided[0, 0, endpoint])

        torch.testing.assert_close(rate_kl, weighted_cross_entropy)

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
