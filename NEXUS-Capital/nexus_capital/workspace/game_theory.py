"""
Latent Game Theory: расчёт равновесий Нэша для переговоров, аукционов
и ценовых войн.

Реализованы:
    • Nash bargaining — деление излишка между продавцом и покупателем;
    • Best-response итерации для матричных игр (2 игрока, дискретные
      стратегии) с поиском ε-Nash равновесия;
    • Ценовая конкуренция Бертрана с эластичностью спроса.
"""
from __future__ import annotations

import torch


def nash_bargaining(
    reservation_a: torch.Tensor,
    reservation_b: torch.Tensor,
    surplus: torch.Tensor,
    bargaining_power_a: torch.Tensor | float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Симметричное решение Неша для двусторонних переговоров.
    Каждая сторона получает свою резервную цену + долю излишка,
    пропорциональную переговорной силе.

    return: (доля A, доля B)
    """
    if isinstance(bargaining_power_a, float):
        pa = torch.full_like(surplus, bargaining_power_a)
    else:
        pa = bargaining_power_a
    share_a = reservation_a + pa * surplus
    share_b = reservation_b + (1.0 - pa) * surplus
    return share_a, share_b


def bertrand_price(
    cost: torch.Tensor,
    competitor_price: torch.Tensor,
    elasticity: torch.Tensor,
    market_size: torch.Tensor,
) -> torch.Tensor:
    """
    Оптимальная цена в однопродуктовой модели Бертрана с логит-спросом.
    Прибыль = (p - cost) * market_size * sigmoid(elasticity*(p_comp - p)).
    Решение ищется градиентным спуском по скаляру p на каждый батч.
    """
    # Отвязываем от внешнего графа, чтобы внутренний backward LBFGS
    # не конфликтовал с основным backward модели.
    cost = cost.detach()
    competitor_price = competitor_price.detach()
    elasticity = elasticity.detach()
    market_size = market_size.detach()

    p = competitor_price.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], lr=0.1, max_iter=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        demand = market_size * torch.sigmoid(elasticity * (competitor_price - p))
        profit = (p - cost) * demand
        loss = -profit.sum()
        loss.backward()
        return loss

    opt.step(closure)
    return p.detach().clamp(min=cost)


def find_nash_equilibrium(
    payoff_a: torch.Tensor,
    payoff_b: torch.Tensor,
    n_iter: int = 200,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fictitious play для двух игроков с дискретными стратегиями.

    payoff_a: (nA, nB) матрица выплат игрока A
    payoff_b: (nA, nB) матрица выплат игрока B
    return: (pA, pB, gap) смешанные стратегии и зазор ε-Nash
    """
    nA, nB = payoff_a.shape
    counts_a = torch.ones(nA, device=payoff_a.device)
    counts_b = torch.ones(nB, device=payoff_b.device)

    for _ in range(n_iter):
        pA = counts_a / counts_a.sum()
        pB = counts_b / counts_b.sum()
        # Лучший ответ A на стратегию B
        exp_a = payoff_a @ pB                 # (nA,)
        exp_b = pA @ payoff_b                 # (nB,)
        br_a = exp_a.argmax()
        br_b = exp_b.argmax()
        counts_a[br_a] += 1
        counts_b[br_b] += 1

    pA = counts_a / counts_a.sum()
    pB = counts_b / counts_b.sum()

    # Зазор равновесия: максимум, сколько можно выиграть отклонением
    value_a = (pA @ payoff_a @ pB)
    value_b = (pA @ payoff_b @ pB)
    dev_a = (payoff_a @ pB).max()
    dev_b = (pA @ payoff_b).max()
    gap = torch.clamp(torch.max(dev_a - value_a, dev_b - value_b), min=0.0)
    return pA, pB, gap
