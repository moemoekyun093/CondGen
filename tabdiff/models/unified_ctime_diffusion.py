import torch.nn.functional as F
import torch
import math
import numpy as np
from tabdiff.models.noise_schedule import *
from tabdiff.models.doob_h_transform import (
    categorical_candidate_log_h,
    guided_categorical_log_probs,
)
from tabdiff.models.harpoon_style import (
    categorical_set_loss,
    interval_relu_loss,
)
from tqdm import tqdm
from itertools import chain

"""
“Our implementation of the continuous-time masked diffusion is inspired by https://arxiv.org/abs/2406.07524's implementation at [https://github.com/kuleshov-group/mdlm], with modifications to support data distributions that include categorical dimensions of different sizes.”
"""

S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

class UnifiedCtimeDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            denoise_fn,
            y_only_model,
            num_timesteps=1000,
            scheduler='power_mean',
            cat_scheduler='log_linear',
            noise_dist='uniform',
            edm_params={},
            noise_dist_params={},
            noise_schedule_params={},
            sampler_params={},
            device=torch.device('cpu'),
            **kwargs
        ):

        super(UnifiedCtimeDiffusion, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device) if len(num_classes)>0 else torch.tensor([]).to(device).int()
        self.mask_index = torch.tensor(self.num_classes).long().to(device)
        self.neg_infinity = -1000000.0 
        self.num_classes_w_mask = tuple(self.num_classes + 1)

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)
        
        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes)+1)
        offsets = np.append([0], offsets)
        self.slices_for_classes_with_mask = []
        for i in range(1, len(offsets)):
            self.slices_for_classes_with_mask.append(np.arange(offsets[i - 1], offsets[i]))

        self._denoise_fn = denoise_fn
        self.y_only_model = y_only_model
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.edm_params = edm_params
        self.noise_dist_params = noise_dist_params
        self.sampler_params = sampler_params
        if self.num_numerical_features == 0:
            self.sampler_params['stochastic_sampler'] = False
            self.sampler_params['second_order_correction'] = False
        
        self.w_num = 0.0
        self.w_cat = 0.0
        self.num_mask_idx = []
        self.cat_mask_idx = []
        self.numerical_h_guide = None
        self.categorical_h_guide = None
        self.h_guide_strength = 1.0
        self.h_guide_max_correction = None
        self.h_guide_max_log_ratio = 10.0
        self.h_guide_candidate_batch_size = 65536
        self.h_guide_query_active_mask = None
        self.h_guide_query_conditioning = None
        self.h_guide_diagnostics_enabled = False
        self._h_guide_correction_chunks = []
        self._h_guide_time_diagnostics = []
        self.harpoon_style_query = None
        self.harpoon_style_strength = 0.2
        
        self.device = device
        
        if self.scheduler == 'power_mean':
            self.num_schedule = PowerMeanNoise(**noise_schedule_params)
        elif self.scheduler == 'power_mean_per_column':
            self.num_schedule = PowerMeanNoise_PerColumn(num_numerical = num_numerical_features, **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.scheduler}-- is not implemented for contiuous data at CTIME ")
        
        if self.cat_scheduler == 'log_linear':
            self.cat_schedule = LogLinearNoise(**noise_schedule_params)
        elif self.cat_scheduler == 'log_linear_per_column':
            self.cat_schedule = LogLinearNoise_PerColumn(num_categories = len(num_classes), **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.cat_scheduler}-- is not implemented for discrete data at CTIME ")

    def mixed_loss(self, x):
        b = x.shape[0]
        device = x.device

        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:].long()
        # Sample noise level
        if self.noise_dist == "uniform_t":
            t = torch.rand(b, device=device, dtype=x_num.dtype)
            t = t[:, None]
            sigma_num = self.num_schedule.total_noise(t)
            sigma_cat = self.cat_schedule.total_noise(t)
            dsigma_cat = self.cat_schedule.rate_noise(t)
        else:
            sigma_num = self.sample_ctime_noise(x)       
            t = self.num_schedule.inverse_to_t(sigma_num)
            while torch.any((t < 0) + (t > 1)):     
                # restrict t to [0,1]
                # this iterative approach is equivalent to sampling from a truncated version of the orignal noise distribution
                invalid_idx = ((t < 0) + (t > 1)).nonzero().squeeze(-1)
                sigma_num[invalid_idx] = self.sample_ctime_noise(x[:len(invalid_idx)])
                t = self.num_schedule.inverse_to_t(sigma_num)
            assert not torch.any((t < 0) + (t > 1))
            sigma_cat = self.cat_schedule.total_noise(t)
        # Convert sigma_cat to the corresponding alpha and move_chance
        alpha = torch.exp(-sigma_cat)
        move_chance = -torch.expm1(-sigma_cat)      # torch.expm1 gives better numertical stability
            
        # Continuous forward diff
        x_num_t = x_num
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = x_num + noise * sigma_num
        
        # Discrete forward diff
        x_cat_t = x_cat
        x_cat_t_soft = x_cat # in the case where x_cat is empty, x_cat_t_soft will have the same shape as x_cat
        if x_cat.shape[1] > 0:
            is_learnable = self.cat_scheduler == 'log_linear_per_column'
            strategy = 'soft'if is_learnable else 'hard'
            x_cat_t, x_cat_t_soft = self.q_xt(x_cat, move_chance, strategy=strategy)

        # Predict orignal data (distribution)
        model_out_num, model_out_cat = self._denoise_fn(   
            x_num_t, x_cat_t_soft,
            t.squeeze(), sigma=sigma_num
        )

        d_loss = torch.zeros((1,)).float()
        c_loss = torch.zeros((1,)).float()

        if x_num.shape[1] > 0:
            c_loss = self._edm_loss(model_out_num, x_num, sigma_num)
        if x_cat.shape[1] > 0:
            logits = self._subs_parameterization(model_out_cat, x_cat_t)    # log normalized probabilities, with the entry mask category being set to -inf
            d_loss = self._absorbed_closs(logits, x_cat, sigma_cat, dsigma_cat)
            
        return d_loss.mean(), c_loss.mean()

    def _section4_start_time_weights(self, fixed_columns, t):
        """Discretized Section 4 posterior q_C(t) over reverse start times.

        A categorical column's reverse reveal time X_i has survival function
        P(X_i > t) = alpha_i(t).  Conditioning on every fixed column in C
        being revealed before every free column in reverse time is equivalent
        to max_{j not in C} X_j < min_{i in C} X_i.  The (unnormalized)
        density of min_{i in C} X_i is therefore

          prod_{j not in C}(1-alpha_j(t))
          * prod_{i in C} alpha_i(t)
          * sum_{i in C} rate_i(t).

        ``t`` is the same evenly spaced grid used by the original sampler, so
        normalizing these density values gives its discrete approximation.
        """
        n_columns = len(self.num_classes)
        fixed_columns = sorted(set(int(column) for column in fixed_columns))
        if not fixed_columns:
            raise ValueError("Section 4 posterior requires at least one fixed categorical column")
        if fixed_columns[0] < 0 or fixed_columns[-1] >= n_columns:
            raise ValueError(
                f"fixed categorical columns must lie in [0, {n_columns - 1}]"
            )

        sigma = self.cat_schedule.total_noise(t)
        rate = self.cat_schedule.rate_noise(t)
        if sigma.shape[-1] == 1 and n_columns > 1:
            sigma = sigma.expand(-1, n_columns)
        if rate.shape[-1] == 1 and n_columns > 1:
            rate = rate.expand(-1, n_columns)
        alpha = torch.exp(-sigma).clamp(min=1e-12, max=1.0 - 1e-12)
        rate = rate.clamp_min(1e-12)

        fixed_mask = torch.zeros(n_columns, dtype=torch.bool, device=t.device)
        fixed_mask[fixed_columns] = True
        log_weights = torch.log(alpha[:, fixed_mask]).sum(dim=1)
        log_weights += torch.log(rate[:, fixed_mask].sum(dim=1))
        if (~fixed_mask).any():
            log_weights += torch.log1p(-alpha[:, ~fixed_mask]).sum(dim=1)
        weights = torch.exp(log_weights - log_weights.max())
        if not torch.isfinite(weights).all() or weights.sum() <= 0:
            raise RuntimeError("Section 4 categorical start-time posterior is not finite")
        return weights / weights.sum()

    def sample_section4_start_indices(self, num_samples, fixed_columns, t):
        """Draw one posterior reverse-start grid index for each trajectory."""
        weights = self._section4_start_time_weights(fixed_columns, t)
        return torch.multinomial(weights, num_samples, replacement=True)

    @torch.no_grad()
    def sample(
        self,
        num_samples,
        fixed_categorical=None,
        categorical_start_mode="full",
    ):
        b = num_samples
        device = self.device
        dtype = torch.float32
        
        # Create the chain of t
        t = torch.linspace(0,1,self.num_timesteps, dtype=dtype, device=device)      # times = 0.0,...,1.0
        t = t[:, None]
        
        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]
        
        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params['stochastic_sampler']:
            gamma = min(S_churn / self.num_timesteps, np.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values    # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma==0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur
                
        fixed_categorical = dict(fixed_categorical or {})
        if categorical_start_mode not in {"full", "section4_posterior"}:
            raise ValueError(
                "categorical_start_mode must be 'full' or 'section4_posterior'"
            )
        for column, value in fixed_categorical.items():
            if column < 0 or column >= len(self.num_classes):
                raise ValueError(f"categorical column index out of range: {column}")
            if value < 0 or value >= int(self.num_classes[column]):
                raise ValueError(
                    f"category {value} is invalid for column {column}; "
                    f"expected [0, {int(self.num_classes[column]) - 1}]"
                )

        start_indices = torch.full(
            (b,), self.num_timesteps - 1, dtype=torch.long, device=device
        )
        if categorical_start_mode == "section4_posterior" and fixed_categorical:
            start_indices = self.sample_section4_start_indices(
                b,
                fixed_categorical.keys(),
                t,
            )
            sampled_t = t[start_indices, 0]
            print(
                "Section 4 categorical start times: "
                f"mean={sampled_t.mean().item():.4f}, "
                f"min={sampled_t.min().item():.4f}, "
                f"max={sampled_t.max().item():.4f}"
            )

        # At the posterior start time, initialize the continuous part at that
        # time's prior scale.  This is the practical mixed-data approximation;
        # Section 4's exact ordering argument itself concerns categorical data.
        start_sigma_num = sigma_num_cur[start_indices]
        z_norm = torch.randn(
            (b, self.num_numerical_features), device=device
        ) * start_sigma_num
            
        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b, 0), device=device).float()      # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            z_cat = self._sample_masked_prior(
                b,
                len(self.num_classes),
            )
            for column, value in fixed_categorical.items():
                z_cat[:, column] = value
        
        largest_start = int(start_indices.max().item())
        pbar = tqdm(reversed(range(0, largest_start + 1)), total=largest_start + 1)
        pbar.set_description(f"Sampling Progress")
        for i in pbar:
            updated_norm, updated_cat, q_xs = self.edm_update(
                z_norm, z_cat, i, 
                t[i], t[i-1] if i > 0 else None, t_hat[i],
                sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
                fixed_categorical=fixed_categorical,
            )
            # Churn can remask an already fixed entry before the MDLM update;
            # restore C_0 so fixed values remain observed for the whole path.
            for column, value in fixed_categorical.items():
                updated_cat[:, column] = value
            active = start_indices >= i
            z_norm = torch.where(active[:, None], updated_norm, z_norm)
            z_cat = torch.where(active[:, None], updated_cat, z_cat)
        
        assert torch.all(z_cat < self.mask_index)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample

    def sample_all(
        self,
        num_samples,
        batch_size,
        keep_nan_samples=False,
        fixed_categorical=None,
        categorical_start_mode="full",
    ):
        b = batch_size

        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f"Samples left to generate: {num_samples-num_generated}")
            sample = self.sample(
                b,
                fixed_categorical=fixed_categorical,
                categorical_start_mode=categorical_start_mode,
            )
            mask_nan = torch.any(sample.isnan(), dim=1)
            if keep_nan_samples:
                # If the sample instances that contains Nan are decided to be kept, the row with Nan will be foreced to all zeros
                sample = sample * (~mask_nan)[:, None]
            else:
                # Otherwise the instances with Nan will be eliminated
                sample = sample[~mask_nan]

            all_samples.append(sample)
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]

        return x_gen
    
    def q_xt(self, x, move_chance, strategy='hard'):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        if strategy == 'hard':
            move_indices = torch.rand(
            * x.shape, device=x.device) < move_chance
            xt = torch.where(move_indices, self.mask_index, x)
            xt_soft = self.to_one_hot(xt).to(move_chance.dtype)
            return xt, xt_soft
        elif strategy == 'soft':
            bs = x.shape[0]
            xt_soft = torch.zeros(bs, torch.sum(self.mask_index+1), device=x.device)
            xt = torch.zeros_like(x)
            for i in range(len(self.num_classes)):
                slice_i = self.slices_for_classes_with_mask[i]
                # set the bernoulli probabilities, which determines the "coin flip" transition to the mask class
                prob_i = torch.zeros(bs, 2, device=x.device)
                prob_i[:,0] = 1-move_chance[:,i]
                prob_i[:,-1] = move_chance[:,i]
                log_prob_i = torch.log(prob_i)
                # draw soft samples and place them back to the corresponding columns
                soft_sample_i = F.gumbel_softmax(log_prob_i, tau=0.01, hard=True)
                idx = torch.stack((x[:,i]+slice_i[0], torch.ones_like(x[:,i])*slice_i[-1]), dim=-1)
                xt_soft[torch.arange(len(idx)).unsqueeze(1), idx] = soft_sample_i
                # retrieve the hard samples
                xt[:, i] = torch.where(soft_sample_i[:,1] > soft_sample_i[:,0], self.mask_index[i], x[:,i])
            return xt, xt_soft
    
    
    def _subs_parameterization(self, unormalized_prob, xt):
        # log prob at the mask index = - infinity
        unormalized_prob = self.pad(unormalized_prob, self.neg_infinity)
        
        unormalized_prob[:, range(unormalized_prob.shape[1]), self.mask_index] += self.neg_infinity
        
        # Take log softmax on the unnormalized probabilities to the logits
        logits = unormalized_prob - torch.logsumexp(unormalized_prob, dim=-1,
                                        keepdim=True)
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)    # (bs, K)
        logits[unmasked_indices] = self.neg_infinity 
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits
    
    def pad(self, x, pad_value):
        """
        Converts a concatenated tensor of class probabilities into a padded matrix, 
        where each sub-tensor is padded along the last dimension to match the largest 
        category size (max number of classes).

        Args:
            x (Tensor): The input tensor containing concatenated probabilities for all the categories in x_cat. 
                        [bs, sum(num_classes_w_mask)]
            pad_value (float): The value filled into the dummy entries, which are padded to ensure all sub-tensors have equal size 
                            along the last dimension.

        Returns:
            Tensor: A new tensorwith
                    [bs, len(num_classes_w_mask), max(num_classes_w_mask)), num_categories]
        """
        splited = torch.split(x, self.num_classes_w_mask, dim=-1)
        max_K = max(self.num_classes_w_mask)
        padded_ = [
            torch.cat((
                t, 
                pad_value*torch.ones(*(t.shape[:-1]), max_K-t.shape[-1], dtype=t.dtype, device=t.device)
            ), dim=-1) 
        for t in splited]
        out = torch.stack(padded_, dim=-2)
        return out
    
    def to_one_hot(self, x_cat):
        x_cat_oh = torch.cat(
            [F.one_hot(x_cat[:, i], num_classes=self.num_classes[i]+1,) for i in range(len(self.num_classes))], 
            dim=-1
        )
        return x_cat_oh
    
    def _absorbed_closs(self, model_output, x0, sigma, dsigma):
        """
            alpha: (bs,)
        """
        log_p_theta = torch.gather(
            model_output, -1, x0[:, :, None]
        ).squeeze(-1)
        alpha = torch.exp(-sigma)
        if self.cat_scheduler in ['log_linear_unified', 'log_linear_per_column']:
            elbo_weight = - dsigma / torch.expm1(sigma)
        else:
            elbo_weight = -1/(1-alpha)
        
        loss = elbo_weight * log_p_theta
        return loss
    
    def _sample_masked_prior(self, *batch_dims):
        return self.mask_index[None,:] * torch.ones(    
        * batch_dims, dtype=torch.int64, device=self.mask_index.device)
        
    def _mdlm_update(
        self,
        log_p_x0,
        x,
        alpha_t,
        alpha_s,
        h_candidate_log_scores=None,
    ):
        """
            # t: (bs,)
            log_p_x0: (bs, K, K_max)
            # alpha_t: (bs,)
            # alpha_s: (bs,)
            alpha_t: (bs, 1/K_cat)
            alpha_s: (bs,1/K_cat)
        """
        # Conditional Generator Matching changes the endpoint-category law,
        # not TabDiff's analytic reveal clock.  Normalize p_base * h before
        # constructing the finite reverse transition, so real-token mass and
        # MASK persistence retain their original schedule-controlled values.
        if h_candidate_log_scores is not None:
            log_p_x0 = guided_categorical_log_probs(
                log_p_x0,
                h_candidate_log_scores,
            )

        move_chance_t = 1 - alpha_t
        move_chance_s = 1 - alpha_s     
        move_chance_t = move_chance_t.unsqueeze(-1)
        move_chance_s = move_chance_s.unsqueeze(-1)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        # There is a noremalizing term is (1-\alpha_t) who's responsility is to ensure q_xs is normalized. 
        # However, omiting it won't make a difference for the Gumbel-max sampling trick in  _sample_categorical()
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, range(q_xs.shape[1]), self.mask_index] = move_chance_s[:, :, 0]
        
        # Important: make sure that prob of dummy classes are exactly 0
        dummy_mask = torch.tensor([[(1 if i <= mask_idx else 0) for i in range(max(self.mask_index+1))] for mask_idx in self.mask_index], device=q_xs.device)
        dummy_mask = torch.ones_like(q_xs) * dummy_mask
        q_xs *= dummy_mask

        _x = self._sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        
        z_cat = copy_flag * x + (1 - copy_flag) * _x
        return copy_flag * x + (1 - copy_flag) * _x, q_xs

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    
    def sample_ctime_noise(self, batch):
        if self.noise_dist == 'log_norm':
            rnd_normal = torch.randn(batch.shape[0], device=batch.device)
            sigma = (rnd_normal * self.noise_dist_params['P_std'] + self.noise_dist_params['P_mean']).exp()
        else:
            raise NotImplementedError(f"The noise distribution--{self.noise_dist}-- is not implemented for CTIME ")
        return sigma

    def set_numerical_h_guide(
        self,
        guide,
        strength=1.0,
        max_correction=None,
        max_log_ratio=10.0,
        candidate_batch_size=65536,
        query_active_mask=None,
    ):
        """Backward-compatible installer for earlier single-guide checkpoints."""
        self.set_doob_h_guides(
            guide,
            guide,
            strength=strength,
            max_correction=max_correction,
            max_log_ratio=max_log_ratio,
            candidate_batch_size=candidate_batch_size,
            query_active_mask=query_active_mask,
        )

    def set_doob_h_guides(
        self,
        numerical_guide,
        categorical_guide,
        strength=1.0,
        max_correction=None,
        max_log_ratio=10.0,
        candidate_batch_size=65536,
        query_active_mask=None,
        query_conditioning=None,
    ):
        """Install separate numerical-score and categorical-log-h guides."""
        if self.harpoon_style_query is not None:
            raise RuntimeError("Doob and HARPOON-style guidance cannot be enabled together")
        if strength < 0:
            raise ValueError("h-guide strength must be non-negative")
        if max_correction is not None and max_correction <= 0:
            raise ValueError("max_correction must be positive when provided")
        if max_log_ratio <= 0 or candidate_batch_size <= 0:
            raise ValueError("max_log_ratio and candidate_batch_size must be positive")
        self.numerical_h_guide = numerical_guide
        self.categorical_h_guide = categorical_guide
        self.h_guide_strength = float(strength)
        self.h_guide_max_correction = max_correction
        self.h_guide_max_log_ratio = float(max_log_ratio)
        self.h_guide_candidate_batch_size = int(candidate_batch_size)
        if query_active_mask is None:
            self.h_guide_query_active_mask = None
        else:
            query_active_mask = torch.as_tensor(
                query_active_mask,
                device=self.device,
                dtype=torch.float32,
            ).reshape(-1)
            if query_active_mask.numel() != self.num_numerical_features:
                raise ValueError(
                    "query_active_mask must contain one entry per numerical column"
                )
            self.h_guide_query_active_mask = query_active_mask
        if query_conditioning is None:
            self.h_guide_query_conditioning = None
        else:
            self.h_guide_query_conditioning = {
                name: torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(-1)
                for name, value in query_conditioning.items()
            }

    def set_harpoon_style_guidance(self, query_conditioning, strength=0.2):
        """Install test-time manifold guidance through the frozen dirty estimate."""
        if strength <= 0:
            raise ValueError("HARPOON-style guidance strength must be positive")
        if self.numerical_h_guide is not None or self.categorical_h_guide is not None:
            raise RuntimeError("Doob and HARPOON-style guidance cannot be enabled together")
        required = {
            "query_lower",
            "query_upper",
            "query_numerical_active",
            "query_categorical_allowed",
            "query_categorical_active",
        }
        missing = required - set(query_conditioning)
        if missing:
            raise ValueError(f"HARPOON-style query is missing fields: {sorted(missing)}")
        query = {
            name: torch.as_tensor(value, device=self.device, dtype=torch.float32)
            .detach()
            .reshape(-1)
            for name, value in query_conditioning.items()
        }
        expected = {
            "query_lower": self.num_numerical_features,
            "query_upper": self.num_numerical_features,
            "query_numerical_active": self.num_numerical_features,
            "query_categorical_allowed": int(np.asarray(self.num_classes).sum()),
            "query_categorical_active": len(self.num_classes),
        }
        for name, width in expected.items():
            if query[name].numel() != width:
                raise ValueError(
                    f"{name} has {query[name].numel()} values; expected {width}"
                )
        allowed_parts = torch.split(
            query["query_categorical_allowed"], self.num_classes.tolist()
        )
        for column, (is_active, allowed) in enumerate(
            zip(query["query_categorical_active"], allowed_parts)
        ):
            if is_active > 0 and allowed.sum() <= 0:
                raise ValueError(f"active categorical column {column} has an empty set")
        self.harpoon_style_query = query
        self.harpoon_style_strength = float(strength)

    def _harpoon_style_prediction(self, x_num_t, x_cat_t_soft, t, sigma):
        """Denoise and differentiate the query loss through the frozen backbone."""
        if self.harpoon_style_query is None:
            denoised, raw_logits = self._denoise_fn(x_num_t, x_cat_t_soft, t, sigma=sigma)
            return denoised, raw_logits, torch.zeros_like(x_num_t)

        batch_size = x_num_t.shape[0]
        query = {
            name: value.to(device=x_num_t.device, dtype=x_num_t.dtype)[None, :]
            .expand(batch_size, -1)
            for name, value in self.harpoon_style_query.items()
        }
        with torch.enable_grad():
            numerical_input = x_num_t.detach().requires_grad_(True)
            categorical_input = x_cat_t_soft.detach().requires_grad_(True)
            denoised, raw_logits = self._denoise_fn(
                numerical_input,
                categorical_input,
                t,
                sigma=sigma,
            )
            loss = interval_relu_loss(
                denoised,
                query["query_lower"],
                query["query_upper"],
                query["query_numerical_active"],
            )
            if len(self.num_classes) > 0:
                loss = loss + categorical_set_loss(
                    raw_logits,
                    query["query_categorical_allowed"],
                    query["query_categorical_active"],
                    self.num_classes.tolist(),
                )
            numerical_gradient, categorical_gradient = torch.autograd.grad(
                loss.sum(),
                (numerical_input, categorical_input),
            )

            # A categorical state cannot be displaced continuously. Transfer the
            # manifold gradient to the real-category logits used by MDLM instead.
            categorical_logit_parts = []
            for indices in self.slices_for_classes_with_mask:
                state_gradient = categorical_gradient[:, indices]
                categorical_logit_parts.append(
                    torch.cat(
                        (
                            state_gradient[:, :-1],
                            torch.zeros_like(state_gradient[:, -1:]),
                        ),
                        dim=1,
                    )
                )
            guided_logits = raw_logits - self.harpoon_style_strength * torch.cat(
                categorical_logit_parts, dim=1
            ) if categorical_logit_parts else raw_logits
            numerical_step = -self.harpoon_style_strength * numerical_gradient

        return denoised.detach(), guided_logits.detach(), numerical_step.detach()

    def _guide_query_mask(self, batch_size, device, dtype):
        if self.h_guide_query_active_mask is None:
            return None
        return self.h_guide_query_active_mask.to(
            device=device,
            dtype=dtype,
        )[None, :].expand(batch_size, -1)

    def _guide_query_kwargs(self, batch_size, device, dtype):
        query_mask = self._guide_query_mask(batch_size, device, dtype)
        kwargs = {"query_active_mask": query_mask}
        if self.h_guide_query_conditioning is not None:
            kwargs.update(
                {
                    name: value.to(device=device, dtype=dtype)[None, :].expand(
                        batch_size, -1
                    )
                    for name, value in self.h_guide_query_conditioning.items()
                }
            )
        return kwargs

    def _guide_state_numerical_scale(self, sigma, reference):
        sigma = torch.as_tensor(
            sigma, device=reference.device, dtype=reference.dtype
        )
        if sigma.ndim == 1:
            sigma = sigma[None, :].expand(reference.shape[0], -1)
        if getattr(self._denoise_fn, "precond", False):
            sigma_data = self._denoise_fn.denoise_fn_D.sigma_data
            return (sigma_data**2 + sigma**2).rsqrt()
        return torch.ones_like(reference)

    def enable_numerical_h_guide_diagnostics(self):
        """Collect raw pre-clipping numerical corrections during sampling."""
        self.h_guide_diagnostics_enabled = True
        self._h_guide_correction_chunks = []
        self._h_guide_time_diagnostics = []

    def numerical_h_guide_diagnostics(self):
        """Summarize correction magnitude and clipping by numerical column."""
        if not self._h_guide_correction_chunks:
            return {
                "enabled": self.h_guide_diagnostics_enabled,
                "num_guide_evaluations": 0,
                "clip_threshold": self.h_guide_max_correction,
                "overall_clip_rate": 0.0,
                "per_column": [],
                "per_time_call": [],
            }
        corrections = torch.cat(self._h_guide_correction_chunks, dim=0)
        threshold = self.h_guide_max_correction
        clipped = (
            corrections.abs() > threshold
            if threshold is not None
            else torch.zeros_like(corrections, dtype=torch.bool)
        )
        per_column = []
        for column in range(corrections.shape[1]):
            values = corrections[:, column]
            absolute = values.abs()
            column_clipped = clipped[:, column]
            per_column.append(
                {
                    "model_index": column,
                    "mean": float(values.mean()),
                    "std": float(values.std(unbiased=False)),
                    "mean_absolute": float(absolute.mean()),
                    "absolute_q50": float(torch.quantile(absolute, 0.50)),
                    "absolute_q90": float(torch.quantile(absolute, 0.90)),
                    "absolute_q99": float(torch.quantile(absolute, 0.99)),
                    "absolute_q999": float(torch.quantile(absolute, 0.999)),
                    "absolute_max": float(absolute.max()),
                    "clip_rate": float(column_clipped.float().mean()),
                    "positive_clip_rate": float(
                        (values > threshold).float().mean()
                        if threshold is not None
                        else 0.0
                    ),
                    "negative_clip_rate": float(
                        (values < -threshold).float().mean()
                        if threshold is not None
                        else 0.0
                    ),
                }
            )
        return {
            "enabled": True,
            "num_guide_evaluations": int(corrections.shape[0]),
            "clip_threshold": threshold,
            "overall_clip_rate": float(clipped.float().mean()),
            "per_column": per_column,
            "per_time_call": self._h_guide_time_diagnostics,
        }

    def _apply_numerical_h_guide(self, denoised, x_num_t, x_cat_t, t, sigma):
        if self.numerical_h_guide is None or self.num_numerical_features == 0:
            return denoised
        query_kwargs = self._guide_query_kwargs(
            x_num_t.shape[0], x_num_t.device, x_num_t.dtype
        )
        if self.h_guide_query_conditioning is not None:
            query_kwargs["state_numerical_scale"] = self._guide_state_numerical_scale(
                sigma, x_num_t
            )
        correction = self.numerical_h_guide(
            x_num_t,
            x_cat_t,
            t,
            **query_kwargs,
        )
        if getattr(self.numerical_h_guide, "scalar_h_gradient", False):
            sigma = torch.as_tensor(
                sigma,
                device=correction.device,
                dtype=correction.dtype,
            )
            if sigma.ndim == 1:
                sigma = sigma[None, :]
            correction = sigma.square() * correction
        if correction.shape != denoised.shape:
            raise ValueError(
                "numerical h-guide correction shape "
                f"{tuple(correction.shape)} does not match denoiser output {tuple(denoised.shape)}"
            )
        if self.h_guide_diagnostics_enabled:
            raw = correction.detach().float().cpu()
            self._h_guide_correction_chunks.append(raw)
            threshold = self.h_guide_max_correction
            clip_rate = (
                float((raw.abs() > threshold).float().mean())
                if threshold is not None
                else 0.0
            )
            time_value = torch.as_tensor(t).detach().float().mean().cpu().item()
            self._h_guide_time_diagnostics.append(
                {
                    "call_index": len(self._h_guide_time_diagnostics),
                    "mean_t": float(time_value),
                    "mean_absolute_correction": float(raw.abs().mean()),
                    "absolute_q99": float(torch.quantile(raw.abs(), 0.99)),
                    "clip_rate": clip_rate,
                }
            )
        if self.h_guide_max_correction is not None:
            correction = correction.clamp(
                min=-self.h_guide_max_correction,
                max=self.h_guide_max_correction,
            )
        return denoised + self.h_guide_strength * correction

    @torch.no_grad()
    def _categorical_h_candidate_scores(self, x_num_t, x_cat_t, t, sigma):
        """Evaluate log h(child) for every masked-column real-token child."""
        guide = self.categorical_h_guide
        if guide is None or not hasattr(guide, "log_h") or len(self.num_classes) == 0:
            return None

        b = x_cat_t.shape[0]
        query_kwargs = self._guide_query_kwargs(
            b, x_num_t.device, x_num_t.dtype
        )
        if self.h_guide_query_conditioning is not None:
            query_kwargs["state_numerical_scale"] = self._guide_state_numerical_scale(
                sigma, x_num_t
            )
        candidate_scores = categorical_candidate_log_h(
            guide,
            x_num_t,
            x_cat_t,
            t,
            self.num_classes,
            self.mask_index,
            self.to_one_hot,
            candidate_batch_size=self.h_guide_candidate_batch_size,
            **query_kwargs,
        )
        return self.h_guide_strength * candidate_scores

    def _edm_loss(self, D_yn, y, sigma):
        weight = (sigma ** 2 + self.edm_params['sigma_data'] ** 2) / (sigma * self.edm_params['sigma_data']) ** 2
    
        target = y
        loss = weight * ((D_yn - target) ** 2)

        return loss
    
    def edm_update(
            self, x_num_cur, x_cat_cur, i, 
            t_cur, t_next, t_hat,
            sigma_num_cur, sigma_num_next, sigma_num_hat, 
            sigma_cat_cur, sigma_cat_next, sigma_cat_hat,
            fixed_categorical=None,
        ):
        """
        i = T-1,...,0
        """
        cfg = self.y_only_model is not None
        
        b = x_num_cur.shape[0]
        has_cat = len(self.num_classes) > 0
        
        # Get x_num_hat by move towards the noise by a small step
        x_num_hat = x_num_cur + (sigma_num_hat ** 2 - sigma_num_cur ** 2).sqrt() * S_noise * torch.randn_like(x_num_cur)
        # Get x_cat_hat
        move_chance = -torch.expm1(sigma_cat_cur - sigma_cat_hat)    # the incremental move change is 1 - alpha_t/alpha_s = 1 - exp(sigma_s - sigma_t)
        x_cat_hat, _ = self.q_xt(x_cat_cur, move_chance) if has_cat else (x_cat_cur, x_cat_cur)
        for column, value in dict(fixed_categorical or {}).items():
            x_cat_hat[:, column] = value

        # Get predictions
        x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_hat.dtype) if has_cat else x_cat_hat
        denoised, raw_logits, harpoon_numerical_step = self._harpoon_style_prediction(
            x_num_hat.float(), x_cat_hat_oh,
            t_hat.squeeze().repeat(b), sigma=sigma_num_hat.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
        )
        
        # Apply cfg updates, if is in cfg mode
        is_bin_class = len(self.num_mask_idx) == 0
        is_learnable = self.scheduler=="power_mean_per_column"
        if cfg:
            if not is_learnable:
                sigma_cond = sigma_num_hat
            else:
                if is_bin_class:
                    sigma_cond = (0.002 ** (1/7) + t_hat * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
                else:
                    sigma_cond = sigma_num_hat[self.num_mask_idx]
            y_num_hat = x_num_hat.float()[:, self.num_mask_idx]
            idx = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]))
            y_cat_hat = x_cat_hat_oh[:,idx]
            y_only_denoised, y_only_raw_logits = self.y_only_model(
                y_num_hat, 
                y_cat_hat,
                t_hat.squeeze().repeat(b), sigma=sigma_cond.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
            )
            
            denoised[:, self.num_mask_idx] *= 1 + self.w_num
            denoised[:, self.num_mask_idx] -= self.w_num*y_only_denoised
            
            mask_logit_idx = [self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]
            mask_logit_idx = np.concatenate(mask_logit_idx) if len(mask_logit_idx)>0 else np.array([])
            
            raw_logits[:, mask_logit_idx] *= 1 + self.w_cat
            raw_logits[:, mask_logit_idx] -= self.w_cat*y_only_raw_logits

        denoised = self._apply_numerical_h_guide(
            denoised,
            x_num_hat.float(),
            x_cat_hat_oh,
            t_hat.squeeze().repeat(b),
            sigma_num_hat.unsqueeze(0).repeat(b, 1),
        )
        
        # Euler step
        d_cur = (x_num_hat - denoised) / sigma_num_hat
        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur
        
        # Unmasking
        x_cat_next = x_cat_cur
        q_xs = torch.zeros_like(x_cat_cur).float()
        if has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b,1)
            alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b,1)
            h_candidate_log_scores = self._categorical_h_candidate_scores(
                x_num_hat.float(),
                x_cat_hat,
                t_hat.squeeze().repeat(b),
                sigma_num_hat.unsqueeze(0).repeat(b, 1),
            )
            x_cat_next, q_xs = self._mdlm_update(
                logits,
                x_cat_hat,
                alpha_t,
                alpha_s,
                h_candidate_log_scores=h_candidate_log_scores,
            )
        
        # Apply 2nd order correction.
        if self.sampler_params['second_order_correction']:
            if i > 0:
                x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_next.dtype) if has_cat else x_cat_hat
                denoised, raw_logits = self._denoise_fn(
                    x_num_next.float(), x_cat_hat_oh,
                    t_next.squeeze().repeat(b), sigma=sigma_num_next.unsqueeze(0).repeat(b,1)
                )
                if cfg:
                    if not is_learnable:
                        sigma_cond = sigma_num_next
                    else:
                        if is_bin_class:
                            sigma_cond = (0.002 ** (1/7) + t_next * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
                        else:
                            sigma_cond = sigma_num_next[self.num_mask_idx]
                    y_num_next = x_num_next.float()[:, self.num_mask_idx]
                    idx = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]))
                    y_cat_hat = x_cat_hat_oh[:, idx]
                    y_only_denoised, y_only_raw_logits = self.y_only_model(
                        y_num_next,
                        y_cat_hat,
                        t_next.squeeze().repeat(b), sigma=sigma_cond.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
                    )
                    denoised[:, self.num_mask_idx] *= 1 + self.w_num
                    denoised[:, self.num_mask_idx] -= self.w_num*y_only_denoised

                denoised = self._apply_numerical_h_guide(
                    denoised,
                    x_num_next.float(),
                    x_cat_hat_oh,
                    t_next.squeeze().repeat(b),
                    sigma_num_next.unsqueeze(0).repeat(b, 1),
                )
                
                d_prime = (x_num_next - denoised) / sigma_num_next
                x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # HARPOON applies its manifold constraint step after the ordinary reverse
        # diffusion update. Categorical coordinates were already transferred to
        # the MDLM endpoint logits above.
        x_num_next = x_num_next + harpoon_numerical_step
        
        return x_num_next, x_cat_next, q_xs


    def sample_impute(self, x_num, x_cat, num_mask_idx, cat_mask_idx, resample_rounds, impute_condition, w_num, w_cat):
        self.w_num = w_num
        self.w_cat = w_cat
        self.num_mask_idx = num_mask_idx
        self.cat_mask_idx = cat_mask_idx
        
        b = x_num.size(0)
        device = self.device
        dtype = torch.float32

        # Create masks, true for the missing columns
        num_mask = [i in num_mask_idx for i in range(self.num_numerical_features)]
        cat_mask = [i in cat_mask_idx for i in range(len(self.num_classes))]
        num_mask = torch.tensor(num_mask).to(x_num.device).to(x_num.dtype)
        cat_mask = torch.tensor(cat_mask).to(x_cat.device).to(x_cat.dtype)

        # Create the chain of t
        t = torch.linspace(0,1,self.num_timesteps, dtype=dtype, device=device)      # times = 0.0,...,1.0
        t = t[:, None]
        
        # Compute the chains of sigma
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]
        
        # Prepare sigma_hat for stochastic sampling mode
        if self.sampler_params['stochastic_sampler']:
            gamma = min(S_churn / self.num_timesteps, np.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values    # take the samllest t_hat induced by sigma_num
            zero_gamma = (gamma==0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat = t
            sigma_num_hat = sigma_num_cur
            sigma_cat_hat = sigma_cat_cur

        # Sample priors for the continuous dimensions
        if impute_condition == "x_t":
            z_norm = x_num + torch.randn((b, self.num_numerical_features), device=device) * sigma_num_cur[-1]   # z_{t_max} = x_0(masked) + sigma_max*epsilon
        elif impute_condition == "x_0":
            z_norm = x_num
            
        # Sample priors for the discrete dimensions
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b, 0), device=device).float()      # the default values for categorical sample if the dataset has no categorical entry
        if has_cat:
            if impute_condition == "x_t":
                z_cat = self._sample_masked_prior(
                    b,
                    len(self.num_classes),
                )   # z_{t_max} is still all pushed to [MASK]
            elif impute_condition == "x_0":
                z_cat = x_cat
        
        pbar = tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps)
        pbar.set_description(f"Sampling Progress")
        for i in pbar:
            for u in range (resample_rounds):
                # Get known parts by Forward Flow
                if impute_condition == "x_t":
                    z_norm_known = x_num + torch.randn((b, self.num_numerical_features), device=device) * sigma_num_next[i]
                    move_chance = 1 - torch.exp(-sigma_cat_next[i]) if i < (self.num_timesteps-1) else torch.ones_like(sigma_cat_next[i])     # force move_chance to be 1 for the first iteration
                    z_cat_known, _ = self.q_xt(x_cat, move_chance)
                elif impute_condition == "x_0":
                    z_norm_known = x_num
                    z_cat_known = x_cat
                
                # Get unknown by Reverse Step
                z_norm_unknown, z_cat_unknown, q_xs = self.edm_update(
                    z_norm, z_cat, i, 
                    t[i], t[i-1] if i > 0 else None, t_hat[i],
                    sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                    sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
                )
                z_norm = (1 - num_mask)  * z_norm_known + num_mask * z_norm_unknown
                z_cat = (1 - cat_mask) * z_cat_known + cat_mask * z_cat_unknown

                # Resample x_t from x_{t-1} by Foward Step
                if u < resample_rounds-1:
                    z_norm = z_norm + (sigma_num_cur[i] ** 2 - sigma_num_next[i] ** 2).sqrt() * S_noise * torch.randn_like(z_norm)
                    move_chance = -torch.expm1(sigma_cat_next[i] - sigma_cat_cur[i])
                    z_cat, _ = self.q_xt(z_cat, move_chance)
        
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample
