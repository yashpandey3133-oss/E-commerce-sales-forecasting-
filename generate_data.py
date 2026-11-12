"""Generate a synthetic e-commerce sales dataset (daily, by category).

Schema (matches the project presentation):
    date, category, units_sold, price, promo_flag, discount_pct, traffic, revenue

Sales(t) = Trend x Seasonality x Promotion effect x Noise
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2023-01-01", "2025-12-31"

# category: (base daily units, average price, festive sensitivity)
CATEGORIES = {
    "Electronics": (55, 420.0, 1.35),
    "Fashion": (80, 55.0, 1.25),
    "Home & Kitchen": (60, 95.0, 1.15),
    "Beauty": (45, 30.0, 1.10),
}

# day-of-week multipliers (Mon..Sun): weekends are stronger
DOW = np.array([0.93, 0.92, 0.95, 0.98, 1.05, 1.18, 1.12])


def seasonal_year(doy: np.ndarray, festive: float) -> np.ndarray:
    """Smooth yearly seasonality with a festive peak in Oct-Nov."""
    base = 1 + 0.06 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    festive_peak = np.exp(-0.5 * ((doy - 305) / 22) ** 2) * 0.45 * festive
    return base + festive_peak


def make_promo_calendar(dates: pd.DatetimeIndex, rng: np.random.Generator):
    """Random 3-5 day campaigns plus fixed festive sale windows each year."""
    promo = np.zeros(len(dates), dtype=int)
    discount = np.zeros(len(dates))
    day = 0
    while day < len(dates):
        if rng.random() < 0.035:
            length = int(rng.integers(3, 6))
            depth = float(rng.choice([10, 15, 20, 25]))
            promo[day:day + length] = 1
            discount[day:day + length] = depth
            day += length + 5
        else:
            day += 1
    for year in sorted(set(dates.year)):
        for start, length, depth in [(f"{year}-10-08", 7, 30.0), (f"{year}-11-24", 5, 35.0)]:
            mask = (dates >= start) & (dates < pd.Timestamp(start) + pd.Timedelta(days=length))
            promo[mask] = 1
            discount[mask] = depth
    return promo, discount


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    t = np.arange(n)
    doy = dates.dayofyear.to_numpy()
    dow = dates.dayofweek.to_numpy()

    promo, discount = make_promo_calendar(dates, rng)
    trend = 1 + 0.00055 * t  # steady growth

    frames = []
    for cat, (base_units, price0, festive) in CATEGORIES.items():
        season = seasonal_year(doy, festive)
        promo_effect = 1 + (discount / 100.0) * 1.1 * promo
        noise = rng.lognormal(mean=0.0, sigma=0.06, size=n)

        units = base_units * trend * season * DOW[dow] * promo_effect * noise
        units = np.maximum(units.round(), 0).astype(int)

        price = price0 * (1 - discount / 100.0) * rng.normal(1.0, 0.01, size=n)
        traffic = (units * rng.normal(38, 3, size=n)).round().astype(int)

        df = pd.DataFrame({
            "date": dates,
            "category": cat,
            "units_sold": units,
            "price": price.round(2),
            "promo_flag": promo,
            "discount_pct": discount,
            "traffic": traffic,
        })
        df["revenue"] = (df["units_sold"] * df["price"]).round(2)
        frames.append(df)

    return (pd.concat(frames, ignore_index=True)
              .sort_values(["date", "category"]).reset_index(drop=True))


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "sales.csv"
    data = generate()
    data.to_csv(out, index=False)
    print(f"Saved {len(data):,} rows to {out}")
