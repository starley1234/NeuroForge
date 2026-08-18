"""
Neural SDE — лёгкий дифференцируемый суррогат рынка для латентного
Монте-Карло. Моделирует эволюцию экономического состояния x(t) как:

    dx = μ(x, t; θ) dt + σ(x, t; θ) dW,

где μ — дрейф (тренд/макро), σ — волатильность (риск). Симулятор
векторизован по путям, что даёт ~10 000 симуляций за миллисекунды.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NeuralSDE(nn.Module):
    def __init__(self, d_value: int, d_hidden: int = 256,
                 n_layers: int = 2):
        super().__init__()
        self.d_value = d_value

        mu_layers = []
        in_d = d_value + 1  # + time
        for _ in range(n_layers):
            mu_layers += [nn.Linear(in_d, d_hidden), nn.SiLU()]
            in_d = d_hidden
        mu_layers += [nn.Linear(d_hidden, d_value)]
        self.mu = nn.Sequential(*mu_layers)

        sigma_layers = []
        in_d = d_value + 1
        for _ in range(n_layers):
            sigma_layers += [nn.Linear(in_d, d_hidden), nn.SiLU()]
            in_d = d_hidden
        sigma_layers += [nn.Linear(d_hidden, d_value)]
        self.sigma = nn.Sequential(*sigma_layers)

        # Начальная инициализация sigma близкой к нулю для стабильности
        for m in self.sigma.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t.expand(*x.shape[:-1], 1)], dim=-1)
        return self.mu(inp)

    def diffusion(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t.expand(*x.shape[:-1], 1)], dim=-1)
        return torch.nn.functional.softplus(self.sigma(inp)) + 1e-4

    def step(self, x: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """Один шаг Эйлера-Маруямы."""
        t_t = torch.full((*x.shape[:-1], 1), t, device=x.device,
                         dtype=x.dtype)
        mu = self.drift(x, t_t)
        sig = self.diffusion(x, t_t)
        dW = torch.randn_like(x) * (dt ** 0.5)
        return x + mu * dt + sig * dW

    def simulate(
        self,
        x0: torch.Tensor,
        horizon: int = 32,
        dt: float = 1.0 / 252,
        paths: int = 10000,
        return_paths: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Векторизованная симуляция `paths` траекторий на `horizon` шагов.

        x0: (B, d_value)
        return:
            paths:  (B, paths, horizon, d_value) если return_paths
            final:  (B, paths, d_value)
        """
        B, D = x0.shape
        device, dtype = x0.device, x0.dtype
        x = x0.unsqueeze(1).expand(B, paths, D).clone()  # (B,P,D)
        all_paths = [x] if return_paths else None
        t = 0.0
        for _ in range(horizon):
            x = self.step(x, t, dt)
            t += dt
            if return_paths:
                all_paths.append(x)

        final = x
        result = {"final": final}
        if return_paths:
            result["paths"] = torch.stack(all_paths, dim=2)  # (B,P,H+1,D)
        return result
