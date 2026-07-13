import math
import torch
import torch.nn.functional as F

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial
from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract, directional_sign_loss, second_order_direction_loss, align_expert_by_peak_shift_after_t_numpy



def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='cosine',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            **kwargs
    ):
        super(Diffusion_TS, self).__init__()

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
                                 n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
                                 max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], **kwargs)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps)  

        assert self.sampling_timesteps <= timesteps
        self.fast_sampling = self.sampling_timesteps < timesteps


        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))


        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def output(self, x, t, a, padding_masks=None):
        trend, season = self.model(x, t, a, padding_masks=padding_masks)
        model_output = trend + season
        return model_output

    def model_predictions(self, x, t, a, clip_x_start=False, padding_masks=None):
        if padding_masks is None:
            padding_masks = torch.ones(x.shape[0], self.seq_length, dtype=bool, device=x.device)
        
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.output(x, t, a, padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def guidance_schedule_scale(self, t, guidance_schedule=None):
        if not guidance_schedule:
            return torch.ones_like(t, dtype=torch.float32)

        mode = guidance_schedule.get('mode', 'none')
        if mode in {None, 'none'}:
            return torch.ones_like(t, dtype=torch.float32)
        if mode not in {'weak_strong_weak', 'wsw'}:
            raise ValueError(f'Unknown guidance schedule mode: {mode}')

        low_scale = float(guidance_schedule.get('low_scale', 0.2))
        start_fraction = float(guidance_schedule.get('start_fraction', 0.2))
        end_fraction = float(guidance_schedule.get('end_fraction', 0.8))
        if not 0.0 <= low_scale <= 1.0:
            raise ValueError('guidance_schedule.low_scale must be in [0, 1].')
        if not 0.0 <= start_fraction < end_fraction <= 1.0:
            raise ValueError('guidance_schedule requires 0 <= start_fraction < end_fraction <= 1.')

        progress = 1.0 - t.float() / max(self.num_timesteps - 1, 1)
        scale = torch.ones_like(progress, dtype=torch.float32)

        if start_fraction > 0:
            warmup = progress < start_fraction
            scale = torch.where(
                warmup,
                low_scale + (1.0 - low_scale) * progress / start_fraction,
                scale,
            )

        if end_fraction < 1:
            cooldown = progress > end_fraction
            cooldown_progress = (progress - end_fraction) / (1.0 - end_fraction)
            scale = torch.where(
                cooldown,
                1.0 - (1.0 - low_scale) * cooldown_progress,
                scale,
            )

        return scale.clamp(min=low_scale, max=1.0)

    def pairwise_draw_l1(self, values, num_draws):
        if num_draws < 2:
            return None
        if values.shape[0] % num_draws != 0:
            raise ValueError(f'Batch size {values.shape[0]} is not divisible by num_draws={num_draws}.')

        num_conditions = values.shape[0] // num_draws
        values = values.reshape(num_draws, num_conditions, -1)
        distances = []
        for draw_i in range(num_draws):
            for draw_j in range(draw_i + 1, num_draws):
                distances.append((values[draw_i] - values[draw_j]).abs().mean(dim=-1))
        return torch.stack(distances, dim=0).mean(dim=0)

    def apply_diversity_regularizer(self, x_pre, x_guided, delta_expert, num_draws=None,
                                    diversity_regularizer=None, x_reference=None):
        if not diversity_regularizer:
            return x_guided

        mode = diversity_regularizer.get('mode', 'none')
        if mode in {None, 'none'}:
            return x_guided
        if mode not in {'retain_pairwise', 'retain_pairwise_shadow'}:
            raise ValueError(f'Unknown diversity regularizer mode: {mode}')
        if num_draws is None or int(num_draws) < 2:
            return x_guided

        num_draws = int(num_draws)
        retain_ratio = float(diversity_regularizer.get('retain_ratio', 0.5))
        alpha = float(diversity_regularizer.get('alpha', 0.1))
        strength = float(diversity_regularizer.get('strength', 1.0))
        eps = float(diversity_regularizer.get('eps', 1.0e-12))
        if retain_ratio < 0:
            raise ValueError('diversity_regularizer.retain_ratio must be >= 0.')
        if strength < 0:
            raise ValueError('diversity_regularizer.strength must be >= 0.')

        if mode == 'retain_pairwise_shadow':
            if x_reference is None:
                return x_guided
            reference_values = x_reference.detach()
        else:
            reference_values = x_pre

        with torch.no_grad():
            reference_distance = self.pairwise_draw_l1(reference_values, num_draws)
        guided_distance = self.pairwise_draw_l1(x_guided, num_draws)
        if reference_distance is None or guided_distance is None:
            return x_guided

        shortfall = F.relu(retain_ratio * reference_distance - guided_distance)
        diversity_loss = (shortfall ** 2).mean()
        grad_div = torch.autograd.grad(diversity_loss, x_guided, retain_graph=False, allow_unused=True)[0]
        if grad_div is None:
            return x_guided

        delta_div = -strength * grad_div
        if alpha >= 0:
            with torch.no_grad():
                max_norm = alpha * delta_expert.detach().norm()
                div_norm = delta_div.detach().norm()
                scale = torch.clamp(max_norm / (div_norm + eps), max=1.0)
            delta_div = delta_div * scale
        return x_guided + delta_div

    def p_mean_variance(self, x, t, a, gt, expf, expc, w_v, w_d, guidance_schedule=None,
                        diversity_regularizer=None, num_guidance_draws=None, diversity_reference=None,
                        clip_denoised=True, gamma=1.0):
        schedule_scale = self.guidance_schedule_scale(t, guidance_schedule).to(x.device)
        guidance_enabled = w_v != 0 or w_d != 0
        if guidance_enabled:
            _, x_start_cond = self.model_predictions(x, t, a)
            _, x_start_uncond = self.model_predictions(x, t, -torch.ones_like(a))
        else:
            with torch.no_grad():
                _, x_start_cond = self.model_predictions(x, t, a)
                _, x_start_uncond = self.model_predictions(x, t, -torch.ones_like(a))
        x_start = (1+gamma) * x_start_cond - gamma * x_start_uncond
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        if guidance_enabled:
            x_start = x_start.clone().detach().requires_grad_(True)
            x_pre = x_start.detach()
            expert_loss_cval = directional_sign_loss(x_start, gt, expc, expf)
            expert_loss_cdir = second_order_direction_loss(x_start, gt, expc, expf)
            grad_cval = torch.autograd.grad(expert_loss_cval, x_start, retain_graph=True)[0]
            grad_cdir = torch.autograd.grad(expert_loss_cdir, x_start, retain_graph=True)[0]
            view_shape = (schedule_scale.shape[0],) + (1,) * (x_start.dim() - 1)
            schedule_scale = schedule_scale.view(view_shape)
            x_guided = x_start - (w_v * schedule_scale) * grad_cval - (w_d * schedule_scale) * grad_cdir
            delta_expert = x_guided.detach() - x_pre
            x_start = self.apply_diversity_regularizer(
                x_pre=x_pre,
                x_guided=x_guided,
                delta_expert=delta_expert,
                num_draws=num_guidance_draws,
                diversity_regularizer=diversity_regularizer,
                x_reference=diversity_reference,
            )
        else:
            x_start = x_start.detach()

        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t: int, a, gt, expf, expc, w_v, w_d, guidance_schedule=None,
                 diversity_regularizer=None, num_guidance_draws=None, diversity_reference=None,
                 noise=None, clip_denoised=True):
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = \
            self.p_mean_variance(
                x=x,
                t=batched_times,
                a=a,
                gt=gt,
                expf=expf,
                expc=expc,
                w_v=w_v,
                w_d=w_d,
                guidance_schedule=guidance_schedule,
                diversity_regularizer=diversity_regularizer,
                num_guidance_draws=num_guidance_draws,
                diversity_reference=diversity_reference,
                clip_denoised=clip_denoised,
            )

        if noise is None:
            noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    # @torch.no_grad()
    def sample(self, shape, a, gt, expf, expc, w_v, w_d, guidance_schedule=None,
               diversity_regularizer=None, num_guidance_draws=None):
        device = self.betas.device
        img = torch.randn(shape, device=device)
        diversity_mode = (diversity_regularizer or {}).get('mode', 'none')
        use_shadow_reference = diversity_mode == 'retain_pairwise_shadow' and (w_v != 0 or w_d != 0)
        img_shadow = img.clone() if use_shadow_reference else None
        if w_v != 0 or w_d != 0:
            expf, expc = align_expert_by_peak_shift_after_t_numpy(
                expf.detach().cpu().numpy(),
                gt.detach().cpu().numpy(),
                expc.detach().cpu().numpy(),
            )
            expf = torch.as_tensor(expf, device=device, dtype=gt.dtype)
            expc = torch.as_tensor(expc, device=device, dtype=gt.dtype)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            noise = torch.randn_like(img) if t > 0 else 0.
            diversity_reference = None
            if use_shadow_reference:
                img_shadow, diversity_reference = self.p_sample(
                    img_shadow,
                    t,
                    a,
                    gt,
                    expf,
                    expc,
                    0.,
                    0.,
                    guidance_schedule=None,
                    diversity_regularizer=None,
                    num_guidance_draws=None,
                    noise=noise,
                )
            img, _ = self.p_sample(
                img,
                t,
                a,
                gt,
                expf,
                expc,
                w_v,
                w_d,
                guidance_schedule=guidance_schedule,
                diversity_regularizer=diversity_regularizer,
                num_guidance_draws=num_guidance_draws,
                diversity_reference=diversity_reference,
                noise=noise,
            )
        return img

    @torch.no_grad()
    def fast_sample(self, shape, clip_denoised=True):
        batch, device, total_timesteps, sampling_timesteps, eta = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img

    def generate_mts(self, a, gt, expf, expc, w_v, w_d, guidance_schedule=None,
                     diversity_regularizer=None, num_guidance_draws=None):
        feature_size, seq_length = self.feature_size, self.seq_length
        batch_size = a.shape[0]
        sample_fn = self.fast_sample if self.fast_sampling else self.sample
        return sample_fn(
            (batch_size, seq_length, feature_size),
            a,
            gt,
            expf,
            expc,
            w_v=w_v,
            w_d=w_d,
            guidance_schedule=guidance_schedule,
            diversity_regularizer=diversity_regularizer,
            num_guidance_draws=num_guidance_draws,
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _train_loss(self, x_start, t, a, weight, target=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        if target is None:
            target = x_start

        x = self.q_sample(x_start=x_start, t=t, noise=noise)  
        model_out = self.output(x, t, a, padding_masks)

        train_loss = self.loss_fn(model_out, target, reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss
        weight = weight.squeeze(-1)
        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        train_loss = train_loss * weight
        return train_loss.mean()

    def forward(self, x, a, weight, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, a=a, weight=weight, **kwargs)

    def return_components(self, x, t: int):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'
        t = torch.tensor([t])
        t = t.repeat(b).to(device)
        x = self.q_sample(x, t)
        trend, season, residual = self.model(x, t, return_res=True)
        return trend, season, residual, x

    def fast_sample_infill(self, shape, target, sampling_timesteps, partial_mask=None, clip_denoised=True, model_kwargs=None):
        batch, device, total_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  
        img = torch.randn(shape, device=device)

        for time, time_next in tqdm(time_pairs, desc='conditional sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(img)

            img = pred_mean + sigma * noise
            img = self.langevin_fn(sample=img, mean=pred_mean, sigma=sigma, t=time_cond,
                                   tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
            target_t = self.q_sample(target, t=time_cond)
            img[partial_mask] = target_t[partial_mask]

        img[partial_mask] = target[partial_mask]

        return img

    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='conditional sampling loop time step', total=self.num_timesteps):
            img = self.p_sample_infill(x=img, t=t, clip_denoised=clip_denoised, target=target,
                                       partial_mask=partial_mask, model_kwargs=model_kwargs)
        
        img[partial_mask] = target[partial_mask]
        return img
    
    def p_sample_infill(
        self,
        x,
        target,
        t: int,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None
    ):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, _ = \
            self.p_mean_variance(x=x, t=batched_times, clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  
        sigma = (0.5 * model_log_variance).exp()
        pred_img = model_mean + sigma * noise

        pred_img = self.langevin_fn(sample=pred_img, mean=model_mean, sigma=sigma, t=batched_times,
                                    tgt_embs=target, partial_mask=partial_mask, **model_kwargs)
        
        target_t = self.q_sample(target, t=batched_times)
        pred_img[partial_mask] = target_t[partial_mask]

        return pred_img

    def langevin_fn(
        self,
        coef,
        partial_mask,
        tgt_embs,
        learning_rate,
        sample,
        mean,
        sigma,
        t,
        coef_=0.
    ):
    
        if t[0].item() < self.num_timesteps * 0.05:
            K = 0
        elif t[0].item() > self.num_timesteps * 0.9:
            K = 3
        elif t[0].item() > self.num_timesteps * 0.75:
            K = 2
            learning_rate = learning_rate * 0.5
        else:
            K = 1
            learning_rate = learning_rate * 0.25

        input_embs_param = torch.nn.Parameter(sample)

        with torch.enable_grad():
            for i in range(K):
                optimizer = torch.optim.Adagrad([input_embs_param], lr=learning_rate)
                optimizer.zero_grad()

                x_start = self.output(x=input_embs_param, t=t)

                if sigma.mean() == 0:
                    logp_term = coef * ((mean - input_embs_param) ** 2 / 1.).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = infill_loss.mean(dim=0).sum()
                else:
                    logp_term = coef * ((mean - input_embs_param)**2 / sigma).mean(dim=0).sum()
                    infill_loss = (x_start[partial_mask] - tgt_embs[partial_mask]) ** 2
                    infill_loss = (infill_loss/sigma.mean()).mean(dim=0).sum()
            
                loss = logp_term + infill_loss
                loss.backward()
                optimizer.step()
                epsilon = torch.randn_like(input_embs_param.data)
                input_embs_param = torch.nn.Parameter((input_embs_param.data + coef_ * sigma.mean().item() * epsilon).detach())

        sample[~partial_mask] = input_embs_param.data[~partial_mask]
        return sample
    

if __name__ == '__main__':
    pass
