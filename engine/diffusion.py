import torch

def linear_beta_schedule(steps, beta_start=1e-4, beta_end=2e-2):
    return torch.linspace(beta_start, beta_end, steps)

class GaussianDiffusion:
    def __init__(self, steps=1000):
        self.steps = steps
        self.betas = linear_beta_schedule(steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, clean, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(clean)
        a = self.alpha_bars.to(clean.device)[t].view(-1, 1, 1, 1, 1)
        noisy = a.sqrt() * clean + (1 - a).sqrt() * noise
        return noisy, noise

    @torch.no_grad()
    def step(self, model, x, text_ids, t):
        betas = self.betas.to(x.device)
        alpha = self.alphas.to(x.device)
        alpha_bar = self.alpha_bars.to(x.device)

        pred_noise = model(x, text_ids, t)
        beta_t = betas[t].view(-1, 1, 1, 1, 1)
        alpha_t = alpha[t].view(-1, 1, 1, 1, 1)
        abar_t = alpha_bar[t].view(-1, 1, 1, 1, 1)

        mean = (x - beta_t / (1 - abar_t).sqrt() * pred_noise) / alpha_t.sqrt()

        if int(t[0]) > 0:
            mean = mean + beta_t.sqrt() * torch.randn_like(x)
        return mean
