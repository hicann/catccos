#!/usr/bin/env python3
"""Deterministic host-side MoE routing helpers without torch dependencies."""


def generate_balanced_expert_idx_list(m, expert_num, top_k, ep):
    """Return ``ep`` rank-local [m, top_k] expert-id lists.

    Flattening ranks/tokens/routes produces a round-robin expert stream.  This
    guarantees unique experts for every token and exact global balance whenever
    the total number of routes is divisible by ``expert_num``.
    """
    for name, value in (("m", m), ("expert_num", expert_num), ("top_k", top_k), ("ep", ep)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if top_k > expert_num:
        raise ValueError(f"top_k ({top_k}) must not exceed expert_num ({expert_num})")

    ranks = []
    for rank in range(ep):
        rank_routes = []
        for token in range(m):
            base = (rank * m + token) * top_k
            rank_routes.append([(base + route) % expert_num for route in range(top_k)])
        ranks.append(rank_routes)
    return ranks
