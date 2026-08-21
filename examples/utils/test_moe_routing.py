#!/usr/bin/env python3
import unittest

from moe_routing import generate_balanced_expert_idx_list


class BalancedRoutingTest(unittest.TestCase):
    def test_four_rank_target_is_exact_and_unique(self):
        routes = generate_balanced_expert_idx_list(512, 16, 6, 4)
        counts = [0] * 16
        for rank in routes:
            for token in rank:
                self.assertEqual(len(token), len(set(token)))
                for expert in token:
                    counts[expert] += 1
        self.assertEqual(counts, [768] * 16)

    def test_is_deterministic(self):
        self.assertEqual(
            generate_balanced_expert_idx_list(17, 12, 6, 1), generate_balanced_expert_idx_list(17, 12, 6, 1)
        )

    def test_nondivisible_routes_differ_by_at_most_one(self):
        routes = generate_balanced_expert_idx_list(7, 5, 3, 2)
        counts = [0] * 5
        for rank in routes:
            for token in rank:
                for expert in token:
                    counts[expert] += 1
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_topk_cannot_exceed_experts(self):
        with self.assertRaises(ValueError):
            generate_balanced_expert_idx_list(8, 4, 5, 1)


if __name__ == "__main__":
    unittest.main()
