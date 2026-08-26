"""
Мультиагентный Self-Play (Теория игр) для Data Flywheel.

Сценарии:
  1. B2B переговоры: Агент-Продавец vs Агент-Покупатель.
     Оба делают предложения цены; раунд завершается при согласии или
     истечении времени. Итог — P&L сделки для каждой стороны.
  2. Маркетмейкер vs Арбитражёр: MM ставит спред, арбитражёр решает,
     атаковать ли его; итог — P&L и риск ликвидности.

Каждый эпизод даёт обучающий кортеж:
    { контекст, последовательность офферов, исход P&L, риск }.
"""
from __future__ import annotations

import numpy as np
import torch


class NegotiationAgent:
    """Эвристический агент B2B-торга с обучаемой стратегией (вектор параметров)."""

    def __init__(self, role: str, reservation: float,
                 aggressiveness: float = 0.5, patience: int = 8,
                 rng: np.random.Generator | None = None):
        self.role = role  # 'seller' | 'buyer'
        self.reservation = reservation
        self.alpha = aggressiveness
        self.patience = patience
        self.rng = rng or np.random.default_rng()

    def initial_offer(self, market_price: float) -> float:
        if self.role == "seller":
            return market_price * (1.0 + 0.3 * self.alpha)
        return market_price * (1.0 - 0.3 * self.alpha)

    def counter(self, offer: float, step: int, total: int) -> float:
        # Линейная концессия к резерву с небольшим шумом
        frac = (step + 1) / total
        if self.role == "seller":
            target = self.reservation + (offer - self.reservation) * (1 - frac)
            return max(self.reservation, target + self.rng.normal(0, 0.02))
        target = self.reservation - (self.reservation - offer) * (1 - frac)
        return min(self.reservation, target + self.rng.normal(0, 0.02))


def run_b2b_negotiation(
    market_price: float = 100.0,
    seller_cost: float = 70.0,
    buyer_value: float = 130.0,
    max_rounds: int = 8,
    seed: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    seller = NegotiationAgent("seller", reservation=seller_cost,
                              aggressiveness=rng.uniform(0.2, 0.9), rng=rng)
    buyer = NegotiationAgent("buyer", reservation=buyer_value,
                             aggressiveness=rng.uniform(0.2, 0.9), rng=rng)

    offers = []
    s_offer = seller.initial_offer(market_price)
    b_offer = buyer.initial_offer(market_price)
    offers.append(("seller", s_offer))
    offers.append(("buyer", b_offer))

    deal_price = None
    for r in range(max_rounds):
        if r % 2 == 0:
            s_offer = seller.counter(b_offer, r, max_rounds)
            offers.append(("seller", s_offer))
            if s_offer <= b_offer:
                deal_price = (s_offer + b_offer) / 2
                break
        else:
            b_offer = buyer.counter(s_offer, r, max_rounds)
            offers.append(("buyer", b_offer))
            if b_offer >= s_offer:
                deal_price = (s_offer + b_offer) / 2
                break

    if deal_price is None:
        return {
            "deal": False,
            "offers": offers,
            "seller_pnl": 0.0,
            "buyer_pnl": 0.0,
            "welfare": 0.0,
        }
    return {
        "deal": True,
        "offers": offers,
        "deal_price": deal_price,
        "seller_pnl": deal_price - seller_cost,
        "buyer_pnl": buyer_value - deal_price,
        "welfare": buyer_value - seller_cost,
    }


class MarketMaker:
    def __init__(self, base_spread: float = 0.02, inventory: float = 0.0):
        self.base_spread = base_spread
        self.inventory = inventory

    def quote(self, mid: float) -> tuple[float, float]:
        skew = 0.1 * self.inventory
        bid = mid * (1 - self.base_spread / 2 - skew)
        ask = mid * (1 + self.base_spread / 2 - skew)
        return bid, ask


class Arbitrageur:
    def __init__(self, edge: float = 0.005, risk_aversion: float = 1.0):
        self.edge = edge
        self.gamma = risk_aversion

    def act(self, bid: float, ask: float, fair_value: float) -> dict:
        # Арбитраж если ask < fair_value*(1-edge) или bid > fair_value*(1+edge)
        if ask < fair_value * (1 - self.edge):
            size = 1.0
            return {"action": "buy", "price": ask, "size": size,
                    "pnl": (fair_value - ask) * size}
        if bid > fair_value * (1 + self.edge):
            size = 1.0
            return {"action": "sell", "price": bid, "size": size,
                    "pnl": (bid - fair_value) * size}
        return {"action": "hold", "price": 0.0, "size": 0.0, "pnl": 0.0}


def run_market_episode(
    n_steps: int = 50, mid0: float = 100.0, seed: int | None = None
) -> dict:
    rng = np.random.default_rng(seed)
    mm = MarketMaker(base_spread=rng.uniform(0.01, 0.05))
    arb = Arbitrageur(edge=rng.uniform(0.002, 0.01))
    mid = mid0
    mm_pnl = 0.0
    arb_pnl = 0.0
    inventory = 0.0
    history = []
    for _ in range(n_steps):
        mid = mid * np.exp(rng.normal(0, 0.01))
        bid, ask = mm.quote(mid)
        action = arb.act(bid, ask, mid)
        if action["action"] == "buy":
            mm_pnl += (ask - mid) * action["size"]
            arb_pnl += action["pnl"]
            inventory -= action["size"]
        elif action["action"] == "sell":
            mm_pnl += (mid - bid) * action["size"]
            arb_pnl += action["pnl"]
            inventory += action["size"]
        mm.inventory = inventory
        history.append((bid, ask, action["action"]))
    return {
        "mm_pnl": mm_pnl,
        "arb_pnl": arb_pnl,
        "final_inventory": inventory,
        "history": history,
    }


def generate_self_play_dataset(
    n_episodes: int = 500, seed: int = 42
) -> list[dict]:
    rng = np.random.default_rng(seed)
    data = []
    for _ in range(n_episodes):
        ep = run_b2b_negotiation(
            market_price=float(rng.lognormal(4.6, 0.3)),
            seller_cost=float(rng.lognormal(4.3, 0.3)),
            buyer_value=float(rng.lognormal(4.9, 0.3)),
            max_rounds=int(rng.integers(4, 12)),
            seed=int(rng.integers(1e9)),
        )
        data.append({"type": "b2b", **ep})
    for _ in range(n_episodes // 2):
        ep = run_market_episode(
            n_steps=int(rng.integers(20, 100)),
            seed=int(rng.integers(1e9)),
        )
        data.append({"type": "market", **ep})
    return data


def negotiation_to_tensor(ep: dict, max_rounds: int = 8) -> torch.Tensor:
    """Складывает последовательность офферов в (max_rounds, 2): [who, price]."""
    arr = np.zeros((max_rounds, 2), dtype=np.float32)
    for i, (who, price) in enumerate(ep["offers"][:max_rounds]):
        arr[i, 0] = 1.0 if who == "seller" else -1.0
        arr[i, 1] = np.log(max(price, 1e-3))
    return torch.from_numpy(arr)
