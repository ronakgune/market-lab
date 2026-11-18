"""Summarize the Stream 4 coverage probe into a CSV map + statistics.

Merges the probed checkpoint (coverage_checkpoint.json) with the
pattern-inferred absences (coverage_pattern_inferred.json: Tue/Thu
dates marked absent after the Mon/Wed/Fri snapshot pattern was proven
on the fully-probed 2020 and 2022 weekday censuses — Tue 0/61,
Thu 0/60 across both windows). Every CSV row carries its basis:
"probed" or "pattern".

Output: data/market-lab/options/coverage_spy.csv and a printed
summary (chain days per year, median gap between chain days within
the Friday sample, stress-window presence).
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

OPTIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "market-lab" / "options"

STRESS_WINDOWS = {
    "2020-02-01/2020-04-30 COVID crash": (date(2020, 2, 1), date(2020, 4, 30)),
    "2022-01-01/2022-11-30 bear year": (date(2022, 1, 1), date(2022, 11, 30)),
    "2024-07-01/2024-08-31 vol spike": (date(2024, 7, 1), date(2024, 8, 31)),
    "2025-03-01/2025-04-30 tariff shock": (date(2025, 3, 1), date(2025, 4, 30)),
}


def main() -> None:
    probed = json.loads((OPTIONS_DIR / "coverage_checkpoint.json").read_text())
    inferred_path = OPTIONS_DIR / "coverage_pattern_inferred.json"
    inferred = json.loads(inferred_path.read_text()) if inferred_path.exists() else {}

    rows: list[tuple[str, int, str]] = []
    for day, n in sorted(probed.items()):
        if n >= 0:
            rows.append((day, n, "probed"))
    for day, n in sorted(inferred.items()):
        if day not in probed or probed[day] < 0:
            rows.append((day, n, "pattern"))
    rows.sort()

    out = OPTIONS_DIR / "coverage_spy.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "weekday", "rows", "present", "basis"])
        for day, n, basis in rows:
            d = date.fromisoformat(day)
            w.writerow([day, d.strftime("%a"), n, int(n > 0), basis])
    print(f"wrote {out} ({len(rows)} dates)")

    present = [date.fromisoformat(d) for d, n, _ in rows if n > 0]
    per_year = Counter(d.year for d in present)
    print("chain days per year (probed+pattern set):", dict(sorted(per_year.items())))
    print("first chain day:", present[0], " last:", present[-1])

    fridays = [d for d in present if d.weekday() == 4]
    gaps = [(b - a).days for a, b in zip(fridays, fridays[1:])]
    print(f"Friday chain days: {len(fridays)}; median gap between Friday "
          f"chain days: {statistics.median(gaps)} days; max gap: {max(gaps)}")

    by_weekday = Counter(d.strftime("%a") for d in present)
    print("present by weekday:", dict(by_weekday))

    probed_by_year: dict[int, list[int]] = defaultdict(list)
    for day, n, basis in rows:
        if basis == "probed":
            probed_by_year[date.fromisoformat(day).year].append(int(n > 0))
    for year in sorted(probed_by_year):
        vals = probed_by_year[year]
        print(f"{year}: probed {len(vals)} dates, present {sum(vals)} "
              f"({100.0 * sum(vals) / len(vals):.0f}%)")

    for name, (lo, hi) in STRESS_WINDOWS.items():
        inside = [d for d in present if lo <= d <= hi]
        print(f"stress {name}: {len(inside)} chain days "
              f"({inside[0]} .. {inside[-1]})" if inside else
              f"stress {name}: NONE")


if __name__ == "__main__":
    main()
