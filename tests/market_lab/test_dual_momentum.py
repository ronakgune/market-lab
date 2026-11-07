import unittest
from datetime import date, timedelta

from market_lab.data import Bar
from market_lab.portfolio_construction import dual_momentum_targets, run_dual_momentum_backtest


def bars_from_prices(prices, start=date(2024, 1, 1)):
    return [
        Bar(start + timedelta(days=i), p, p * 1.01, p * 0.99, p, 1_000_000)
        for i, p in enumerate(prices)
    ]


def linear_series(start, step, n=180):
    return [start + step * i for i in range(n)]


class DualMomentumTests(unittest.TestCase):
    def test_dual_momentum_filters_negative_absolute_momentum(self):
        bars_by_symbol = {
            "UP": bars_from_prices(linear_series(100, 0.40)),
            "DOWN": bars_from_prices(linear_series(200, -0.30)),
            "FLAT": bars_from_prices(linear_series(100, 0.0)),
        }

        targets = dual_momentum_targets(
            bars_by_symbol,
            formation_days=90,
            skip_days=0,
            top_n=2,
            absolute_threshold=0.01,
        )

        self.assertEqual([t.symbol for t in targets], ["UP"])
        self.assertGreater(targets[0].absolute_momentum, 0.01)
        self.assertAlmostEqual(targets[0].target_weight, 0.2)
        self.assertEqual(targets[0].rank, 1)

    def test_dual_momentum_selects_relative_winners_and_caps_weights(self):
        bars_by_symbol = {
            "FAST": bars_from_prices(linear_series(100, 0.80)),
            "MID": bars_from_prices(linear_series(100, 0.45)),
            "SLOW": bars_from_prices(linear_series(100, 0.10)),
        }

        targets = dual_momentum_targets(
            bars_by_symbol,
            formation_days=90,
            skip_days=0,
            top_n=2,
            absolute_threshold=0.0,
            max_weight=0.40,
        )

        self.assertEqual([t.symbol for t in targets], ["FAST", "MID"])
        self.assertTrue(all(t.target_weight <= 0.40 for t in targets))
        self.assertAlmostEqual(sum(t.target_weight for t in targets), 0.80)
        self.assertGreater(targets[0].relative_score, targets[1].relative_score)

    def test_dual_momentum_backtest_rebalances_monthly_at_next_open(self):
        bars_by_symbol = {
            "LEADER": bars_from_prices([100 + i for i in range(90)]),
            "LAGGARD": bars_from_prices([100 + 0.1 * i for i in range(90)]),
        }
        # Force a visible next-open fill: first rebalance decision at bar 40 should fill at bar 41 open.
        leader = list(bars_by_symbol["LEADER"])
        leader[41] = Bar(leader[41].date, 200.0, 202.0, 198.0, 201.0, 1_000_000)
        bars_by_symbol["LEADER"] = leader

        result = run_dual_momentum_backtest(
            bars_by_symbol,
            formation_days=20,
            skip_days=0,
            top_n=1,
            rebalance_every=21,
            start_index=40,
            initial_cash=10_000.0,
            max_weight=1.0,
        )

        self.assertGreaterEqual(result.rebalances, 2)
        self.assertEqual(result.snapshots[0].decision_index, 40)
        self.assertEqual(result.snapshots[0].fill_index, 41)
        self.assertEqual(result.snapshots[0].targets[0].symbol, "LEADER")
        # Costs are ON by default since the Stream 2 fix: the recorded fill
        # price is the slipped buy price (200.00 open + 5bps), not the raw open.
        self.assertAlmostEqual(result.snapshots[0].fill_prices["LEADER"], 200.10)
        self.assertGreater(result.final_equity, 0)

    def test_dual_momentum_backtest_handles_empty_series_without_crashing(self):
        result = run_dual_momentum_backtest({"AAPL": [], "MSFT": []})

        self.assertEqual(result.rebalances, 0)
        self.assertEqual(result.final_equity, 10_000.0)

    def test_dual_momentum_backtest_accepts_lowercase_symbols(self):
        bars_by_symbol = {
            "leader": bars_from_prices([100 + i for i in range(90)]),
            "laggard": bars_from_prices([100 + 0.1 * i for i in range(90)]),
        }

        result = run_dual_momentum_backtest(
            bars_by_symbol,
            formation_days=20,
            skip_days=0,
            top_n=1,
            rebalance_every=21,
            start_index=40,
            max_weight=1.0,
        )

        self.assertGreater(result.rebalances, 0)
        self.assertEqual(result.snapshots[0].targets[0].symbol, "LEADER")

    def test_dual_momentum_absolute_filter_uses_current_close_not_skipped_rank_window(self):
        # OLD_WINNER has strong pre-skip relative momentum, then collapses during skipped window.
        old_winner = [100 + i for i in range(70)] + [170 - i * 6 for i in range(21)]
        steady = [100 + i * 0.25 for i in range(91)]
        bars_by_symbol = {
            "OLD_WINNER": bars_from_prices(old_winner),
            "STEADY": bars_from_prices(steady),
        }

        targets = dual_momentum_targets(
            bars_by_symbol,
            formation_days=60,
            skip_days=20,
            top_n=2,
            absolute_threshold=0.0,
            decision_index=90,
        )

        self.assertEqual([t.symbol for t in targets], ["STEADY"])
        self.assertGreater(targets[0].absolute_momentum, 0.0)

    def test_dual_momentum_excludes_symbols_without_requested_decision_bar(self):
        bars_by_symbol = {
            "STALE": bars_from_prices([100 + i for i in range(80)]),
            "CURRENT": bars_from_prices([100 + i * 0.2 for i in range(100)]),
        }

        targets = dual_momentum_targets(
            bars_by_symbol,
            formation_days=30,
            skip_days=0,
            top_n=2,
            absolute_threshold=0.0,
            decision_index=99,
        )

        self.assertEqual([t.symbol for t in targets], ["CURRENT"])

    def test_dual_momentum_backtest_aligns_mixed_history_by_common_dates(self):
        old_start = date(2023, 1, 1)
        new_start = date(2023, 3, 1)
        long_prices = [100 + i * 2 for i in range(59)] + [200 + i * 0.1 for i in range(61)]
        short_prices = [100 + i for i in range(61)]
        bars_by_symbol = {
            "LONG": bars_from_prices(long_prices, start=old_start),
            "SHORT": bars_from_prices(short_prices, start=new_start),
        }

        result = run_dual_momentum_backtest(
            bars_by_symbol,
            formation_days=20,
            skip_days=0,
            top_n=1,
            rebalance_every=21,
            start_index=40,
            max_weight=1.0,
        )

        self.assertGreater(result.rebalances, 0)
        first = result.snapshots[0]
        self.assertEqual(bars_by_symbol["LONG"][-61 + first.fill_index].date, bars_by_symbol["SHORT"][first.fill_index].date)
        self.assertEqual(first.targets[0].symbol, "SHORT")

    def test_dual_momentum_weights_by_selected_count_then_applies_cap(self):
        bars_by_symbol = {
            "A": bars_from_prices(linear_series(100, 1.0)),
            "B": bars_from_prices(linear_series(100, 0.8)),
            "C": bars_from_prices(linear_series(200, -0.5)),
        }

        targets = dual_momentum_targets(
            bars_by_symbol,
            formation_days=60,
            skip_days=0,
            top_n=5,
            absolute_threshold=0.0,
            max_weight=0.30,
        )

        self.assertEqual([t.symbol for t in targets], ["A", "B"])
        self.assertTrue(all(abs(t.target_weight - 0.30) < 1e-9 for t in targets))

    def test_dual_momentum_filters_symbols_below_spy_momentum(self):
        bars_by_symbol = {
            "LEADER": bars_from_prices(linear_series(100, 0.60, n=180)),
            "LAGGER": bars_from_prices(linear_series(100, 0.15, n=180)),
        }
        spy_bars = bars_from_prices(linear_series(100, 0.30, n=180))

        targets = dual_momentum_targets(
            bars_by_symbol,
            formation_days=90,
            skip_days=0,
            top_n=3,
            absolute_threshold=0.0,
            max_weight=0.40,
            spy_bars=spy_bars,
        )

        self.assertEqual([t.symbol for t in targets], ["LEADER"])
        self.assertIn("benchmark/absolute filter", targets[0].reason)


if __name__ == "__main__":
    unittest.main()
