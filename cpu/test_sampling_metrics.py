"""Unit tests for sampling pipeline and metric helpers."""

from __future__ import annotations

import unittest

import numpy as np

from gpt2 import softmax
from metrics import percentile, summarize_latencies_ms
from sampling import (
    SamplingConfig,
    apply_top_k,
    apply_top_p,
    choose_next_token,
    greedy_token_id,
    prepare_logits_for_sampling,
)


class TestGreedyEquivalence(unittest.TestCase):
    def test_temperature_zero_is_argmax(self) -> None:
        logits = np.array([[0.1, 2.0, 1.0]], dtype=np.float32)
        cfg = SamplingConfig(temperature=0.0)
        tok = choose_next_token(logits, [1, 2, 3], cfg, None)
        self.assertEqual(tok, 1)
        self.assertEqual(tok, greedy_token_id(logits[-1]))

    def test_temperature_zero_ignores_top_k_in_prepare(self) -> None:
        # choose_next_token uses raw logits when greedy
        logits = np.array([[10.0, 9.0, 8.0]], dtype=np.float32)
        cfg = SamplingConfig(temperature=0.0, top_k=1)
        self.assertEqual(choose_next_token(logits, [], cfg, None), 0)


class TestTopK(unittest.TestCase):
    def test_top_k_masks_low_logits(self) -> None:
        logits = np.array([1.0, 5.0, 3.0, 2.0])
        out = apply_top_k(logits, 2)
        self.assertTrue(np.isfinite(out[1]))
        self.assertTrue(np.isfinite(out[2]))
        self.assertFalse(np.isfinite(out[0]))
        self.assertFalse(np.isfinite(out[3]))


class TestTopP(unittest.TestCase):
    def test_top_p_keeps_nucleus(self) -> None:
        logits = np.array([0.0, 0.0, 0.0, 10.0], dtype=np.float64)
        out = apply_top_p(logits, 0.5)
        finite = int(np.isfinite(out).sum())
        self.assertEqual(finite, 1)


class TestDeterministicSampling(unittest.TestCase):
    def test_same_seed_same_sample(self) -> None:
        logits = np.array([[1.0, 2.0, 3.0, 1.5]], dtype=np.float32)
        cfg = SamplingConfig(temperature=1.0, top_k=0, top_p=1.0, seed=42)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a = choose_next_token(logits, [0], cfg, rng1)
        b = choose_next_token(logits, [0], cfg, rng2)
        self.assertEqual(a, b)

    def test_temperature_scales_logits(self) -> None:
        raw = np.array([2.0, 4.0, 6.0])
        cfg = SamplingConfig(temperature=2.0)
        proc = prepare_logits_for_sampling(raw[np.newaxis, :], [], cfg)
        np.testing.assert_allclose(proc, raw / 2.0)


class TestValidation(unittest.TestCase):
    def test_invalid_temperature(self) -> None:
        with self.assertRaises(ValueError):
            SamplingConfig(temperature=-0.1).validate()

    def test_invalid_top_p(self) -> None:
        with self.assertRaises(ValueError):
            SamplingConfig(top_p=0.0).validate()


class TestPercentiles(unittest.TestCase):
    def test_percentile_median(self) -> None:
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 50), 2.5)

    def test_summarize_decode(self) -> None:
        s = summarize_latencies_ms([100.0, 200.0, 300.0])
        self.assertAlmostEqual(s["mean_ms"], 200.0)
        self.assertAlmostEqual(s["p50_ms"], 200.0)
        self.assertAlmostEqual(s["min_ms"], 100.0)
        self.assertAlmostEqual(s["max_ms"], 300.0)


class TestGreedyGenerateParity(unittest.TestCase):
    def test_greedy_matches_greedy_generate(self) -> None:
        from pathlib import Path

