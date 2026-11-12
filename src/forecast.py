"""E-commerce Sales Forecasting - Project 50.

Pipeline: Collect -> Clean -> Explore -> Engineer features -> Model -> Evaluate

Models compared on a time-based hold-out (last 90 days):
    1. Seasonal Naive      (baseline)
    2. Holt-Winters        (exponential smoothing)
    3. SARIMAX + Promo     (statistical, promotion flag as exogenous input)
    4. Gradient Boosting   (machine learning on calendar, promo and last-year features)

Usage:
    python forecast.py                      # uses sales.csv
    python forecast.py --data my_sales.csv  # your own dataset (same schema)
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT  # all files live in one folder
HOLDOUT_DAYS = 90
SEASON = 7  # weekly seasonality
LAG = 364   # same weekday, one year ago (>= horizon, so always known)

TEAL, AMBER, GREY = "#0F5C63", "#F2A33A", "#8A9399"


# ----------------------------------------------------------------- 1-2. Collect & clean
def load_and_clean(path: Path) -> pd.DataFrame:
    """Load rows, sort by time, aggregate all categories to one daily series, fill gaps."""
    raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date")

    daily = raw.groupby("date").agg(
        revenue=("revenue", "sum"),
        units_sold=("units_sold", "sum"),
        traffic=("traffic", "sum"),
        promo_flag=("promo_flag", "max"),
        discount_pct=("discount_pct", "max"),
    )
    # fill missing dates so the series has a strict daily index
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    daily[["promo_flag", "discount_pct"]] = daily[["promo_flag", "discount_pct"]].fillna(0)
    daily[["revenue", "units_sold", "traffic"]] = daily[["revenue", "units_sold", "traffic"]].interpolate()

    # cap outliers / stock-out days (winsorise at 1st and 99th percentile of revenue)
    lo, hi = daily["revenue"].quantile([0.01, 0.99])
    daily["revenue"] = daily["revenue"].clip(lo, hi)
    daily.index.name = "date"
    return daily


# ----------------------------------------------------------------- 4. Feature engineering
def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["day_of_week"] = df.index.dayofweek
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["trend_index"] = np.arange(len(df))
    df["last_year_sales"] = df["revenue"].shift(LAG)
    return df


FEATURES = ["last_year_sales", "promo_flag", "discount_pct", "day_of_year",
            "day_of_week", "month", "trend_index"]
FEATURE_LABELS = {
    "last_year_sales": "Last-year sales", "promo_flag": "Promotion flag",
    "discount_pct": "Discount depth", "day_of_year": "Day of year",
    "day_of_week": "Day of week", "month": "Month", "trend_index": "Trend index",
}


# ----------------------------------------------------------------- 5. Models
def seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    last_week = train.iloc[-SEASON:].to_numpy()
    return np.tile(last_week, int(np.ceil(horizon / SEASON)))[:horizon]


def holt_winters(train: pd.Series, horizon: int) -> np.ndarray:
    model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                 seasonal_periods=SEASON).fit()
    return model.forecast(horizon).to_numpy()


def sarimax_promo(train: pd.Series, train_exog: pd.DataFrame,
                  test_exog: pd.DataFrame) -> np.ndarray:
    model = SARIMAX(train, exog=train_exog, order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, SEASON),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=200)
    return fit.forecast(steps=len(test_exog), exog=test_exog).to_numpy()


def gradient_boosting(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.dropna(subset=["last_year_sales"])  # first year has no lag
    model = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                      max_depth=3, subsample=0.8, random_state=42)
    model.fit(train_df[FEATURES], train_df["revenue"])
    return model.predict(test_df[FEATURES]), model


# ----------------------------------------------------------------- 6. Evaluation
def metrics(actual: np.ndarray, forecast: np.ndarray) -> dict:
    err = actual - forecast
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE (%)": float(np.mean(np.abs(err) / actual) * 100),
    }


# ----------------------------------------------------------------- Charts
def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def plot_monthly(daily: pd.DataFrame):
    monthly = daily["revenue"].resample("MS").sum() / 1000
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [GREY, TEAL, AMBER]
    for color, (year, grp) in zip(colors, monthly.groupby(monthly.index.year)):
        ax.plot(grp.index.month, grp.values, marker="o", color=color, label=str(year), lw=2)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_title("Monthly sales (thousands): trend + yearly seasonality")
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (thousands)")
    ax.legend(frameon=False); style_axes(ax)
    fig.tight_layout(); fig.savefig(OUT / "01_monthly_sales.png", dpi=150); plt.close(fig)


def plot_comparison(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [GREY, GREY, GREY, AMBER]
    bars = ax.bar(results.index, results["MAPE (%)"], color=colors[:len(results)])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title(f"MAPE on {HOLDOUT_DAYS}-day hold-out (lower is better)")
    ax.set_ylabel("MAPE (%)"); style_axes(ax)
    plt.setp(ax.get_xticklabels(), rotation=12)
    fig.tight_layout(); fig.savefig(OUT / "02_model_comparison.png", dpi=150); plt.close(fig)


def plot_forecast(train: pd.Series, test: pd.Series, forecasts: dict, best: str):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    tail = train.iloc[-60:]
    ax.plot(tail.index, tail.values, color=GREY, lw=1.5, label="Train (last 60 days)")
    ax.plot(test.index, test.values, color="#333333", lw=1.5, label="Actual")
    ax.plot(test.index, forecasts[best], color=AMBER, lw=2, label=f"Forecast: {best}")
    ax.axvline(test.index[0], color="k", ls="--", lw=0.8)
    ax.set_title("Forecast vs actual on unseen data")
    ax.set_ylabel("Daily revenue"); ax.legend(frameon=False); style_axes(ax)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(OUT / "03_forecast_vs_actual.png", dpi=150); plt.close(fig)


def plot_importance(model: GradientBoostingRegressor):
    imp = pd.Series(model.feature_importances_ * 100, index=FEATURES).sort_values()
    imp.index = [FEATURE_LABELS[i] for i in imp.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(imp.index, imp.values, color=TEAL)
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set_title("Gradient Boosting: feature importance (%)")
    style_axes(ax); ax.grid(axis="x", alpha=0.25); ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(OUT / "04_feature_importance.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------- Main
def main(data_path: Path):
    daily = load_and_clean(data_path)
    print(f"Loaded {len(daily):,} days: {daily.index.min().date()} to {daily.index.max().date()}")
    plot_monthly(daily)

    feat = make_features(daily)
    # Golden rule: split by time, never shuffle.
    train_df, test_df = feat.iloc[:-HOLDOUT_DAYS], feat.iloc[-HOLDOUT_DAYS:]
    train, test = train_df["revenue"], test_df["revenue"]
    print(f"Train: {len(train)} days ({train.index.min().date()} to {train.index.max().date()})")
    print(f"Test : {len(test)} days ({test.index.min().date()} to {test.index.max().date()})\n")

    forecasts = {
        "Seasonal Naive": seasonal_naive(train, HOLDOUT_DAYS),
        "Holt-Winters": holt_winters(train, HOLDOUT_DAYS),
        "SARIMAX + Promo": sarimax_promo(train, train_df[["promo_flag"]], test_df[["promo_flag"]]),
    }
    forecasts["Gradient Boosting"], gb_model = gradient_boosting(train_df, test_df)

    results = pd.DataFrame({name: metrics(test.to_numpy(), f) for name, f in forecasts.items()}).T
    results = results.round(2)
    print(results.to_string())

    best = results["MAPE (%)"].idxmin()
    base = results.loc["Seasonal Naive"]
    gain = (1 - results.loc[best, "MAPE (%)"] / base["MAPE (%)"]) * 100
    print(f"\nBest model: {best} (MAPE {results.loc[best, 'MAPE (%)']:.1f}%, "
          f"{gain:.0f}% lower MAPE than Seasonal Naive)")

    results.to_csv(OUT / "metrics.csv")
    pd.DataFrame({"actual": test, **{k: v for k, v in forecasts.items()}}).round(2) \
        .to_csv(OUT / "forecasts.csv", index_label="date")
    plot_comparison(results)
    plot_forecast(train, test, forecasts, best)
    plot_importance(gb_model)
    print(f"\nCharts and metrics saved to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-commerce sales forecasting")
    parser.add_argument("--data", type=Path, default=ROOT / "sales.csv")
    main(parser.parse_args().data)
