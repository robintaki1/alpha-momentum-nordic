from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_momentum_scores(
    prices: np.ndarray,
    lookback: int,
    skip: int,
) -> np.ndarray:
    """Return simple lookback/skip momentum scores from a 2D price matrix.

    Rows are securities and columns are ordered oldest -> newest.
    """
    array = np.asarray(prices, dtype=float)
    if array.ndim != 2:
        raise ValueError("prices must be a 2D array.")
    required_columns = lookback + skip + 1
    if array.shape[1] < required_columns:
        raise ValueError("prices does not contain enough columns for the requested lookback/skip.")

    current_index = array.shape[1] - skip - 1
    previous_index = current_index - lookback
    current = array[:, current_index]
    previous = array[:, previous_index]
    if np.any(current <= 0.0) or np.any(previous <= 0.0):
        raise ValueError("Momentum scores require strictly positive prices.")
    return current / previous - 1.0


def select_top_n(scores: np.ndarray, eligible: np.ndarray, top_n: int) -> list[int]:
    score_array = np.asarray(scores, dtype=float)
    eligible_array = np.asarray(eligible, dtype=bool)
    if score_array.ndim != 1 or eligible_array.ndim != 1:
        raise ValueError("scores and eligible must be 1D arrays.")
    if score_array.shape[0] != eligible_array.shape[0]:
        raise ValueError("scores and eligible must have the same length.")
    if top_n < 0:
        raise ValueError("top_n must be non-negative.")

    ranked = [index for index, flag in enumerate(eligible_array) if flag]
    ranked.sort(key=lambda index: (-score_array[index], index))
    return ranked[:top_n]


def equal_weight_positions(selected_indices: Iterable[int], universe_size: int) -> np.ndarray:
    if universe_size < 0:
        raise ValueError("universe_size must be non-negative.")
    weights = np.zeros(universe_size, dtype=float)
    selected = list(selected_indices)
    if not selected:
        return weights
    unique = sorted(set(selected))
    weight = 1.0 / len(unique)
    for index in unique:
        if index < 0 or index >= universe_size:
            raise IndexError("selected index is out of range for universe_size.")
        weights[index] = weight
    return weights
