from __future__ import annotations

from typing import Any, Iterable

import numpy as np

import config


def resolve_strategy(params: dict[str, Any]) -> dict[str, Any]:
    strategy_id = params.get("strategy_id") or "baseline"
    base = next(
        (item for item in config.STRATEGY_VARIANTS if item.get("strategy_id") == strategy_id),
        {"strategy_id": strategy_id},
    )
    merged = dict(base)
    for key, value in params.items():
        if key in ("l", "skip", "top_n"):
            continue
        merged[key] = value
    merged.setdefault("strategy_id", strategy_id)
    merged.setdefault("rebalance", "full")
    merged.setdefault("weighting", "equal")
    return merged


def _rankable_order(rank_order: Iterable[int], rankable_mask: np.ndarray) -> tuple[list[int], dict[int, int]]:
    order: list[int] = []
    rank_pos: dict[int, int] = {}
    for security_index in rank_order:
        idx = int(security_index)
        if not rankable_mask[idx]:
            continue
        rank_pos[idx] = len(order) + 1
        order.append(idx)
    return order, rank_pos


def select_indices(
    *,
    rank_order: Iterable[int],
    rankable_mask: np.ndarray,
    top_n: int,
    previous_selection: set[int],
    strategy: dict[str, Any],
    holding_ages: dict[int, int] | None = None,
) -> tuple[list[int], dict[int, int]]:
    rebalance = strategy.get("rebalance", "full")
    band_buffer = int(strategy.get("band_buffer", 0) or 0)
    min_hold = int(strategy.get("min_hold_months", 0) or 0)
    order, rank_pos = _rankable_order(rank_order, rankable_mask)
    if rebalance == "banded":
        kept = [
            index
            for index in previous_selection
            if index in rank_pos and rank_pos[index] <= (top_n + band_buffer)
        ]
        selected = kept + [idx for idx in order if idx not in kept][: max(0, top_n - len(kept))]
        return selected, holding_ages or {}
    if rebalance == "min_hold":
        ages: dict[int, int] = {}
        if holding_ages:
            for index in previous_selection:
                if rankable_mask[index]:
                    ages[index] = holding_ages.get(index, 0) + 1
        locked = [index for index, age in ages.items() if age < max(min_hold, 1)]
        selected = locked + [idx for idx in order if idx not in locked][: max(0, top_n - len(locked))]
        for index in selected:
            if index not in ages:
                ages[index] = 1
        return selected, ages
    selected = order[:top_n]
    return selected, holding_ages or {}


def compute_weights(
    *,
    selected_indices: list[int],
    top_n: int,
    weighting: str,
    weight_values: np.ndarray | None,
) -> dict[int, float]:
    if not selected_indices:
        return {}
    exposure = float(len(selected_indices)) / float(max(1, top_n))
    if weighting == "cap" and weight_values is not None:
        raw = np.array([float(weight_values[index]) for index in selected_indices], dtype=float)
        raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, np.nan)
    elif weighting == "inv_vol" and weight_values is not None:
        raw = np.array([float(weight_values[index]) for index in selected_indices], dtype=float)
        raw = np.where(np.isfinite(raw) & (raw > 0.0), 1.0 / raw, np.nan)
    else:
        raw = np.ones(len(selected_indices), dtype=float)
    if not np.isfinite(raw).any() or np.nansum(raw) <= 0.0:
        raw = np.ones(len(selected_indices), dtype=float)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    total = float(raw.sum())
    if total <= 0.0:
        raw = np.ones(len(selected_indices), dtype=float)
        total = float(raw.sum())
    weights = exposure * raw / total
    return {index: float(weight) for index, weight in zip(selected_indices, weights)}
