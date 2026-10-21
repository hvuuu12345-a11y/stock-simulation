"""Các phân tích bằng pandas cho câu hỏi 1-5."""
import numpy as np
import pandas as pd

from .data import SESSIONS


def long_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Dạng dài: mỗi dòng là một mức giá (365 x 7 dòng)."""
    out = df[SESSIONS].reset_index().melt(id_vars="date", value_name="price")
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    return out.sort_values(["date", "variable"]).reset_index(drop=True)


def _stats(long: pd.DataFrame, key: str) -> pd.DataFrame:
    g = long.groupby(key)["price"]
    t = g.agg(mean="mean", std="std", min="min", max="max")
    t["range"] = t["max"] - t["min"]
    return t.round(2)


def monthly_stats(df):
    return _stats(long_prices(df), "month")


def quarterly_stats(df):
    t = _stats(long_prices(df), "quarter")
    t["cv_%"] = (t["std"] / t["mean"] * 100).round(2)   # hệ số biến thiên
    return t


def abnormal_days(df: pd.DataFrame) -> pd.DataFrame:
    """Ngày có đóng cửa chạm sàn/trần hoặc có giá bằng giá tham chiếu."""
    p = df[SESSIONS]
    ref, fl, ce = df["ref"].to_numpy()[:, None], df["floor"].to_numpy()[:, None], df["ceil"].to_numpy()[:, None]
    n_ceil = np.isclose(p, ce, atol=1e-4).sum(axis=1)
    n_floor = np.isclose(p, fl, atol=1e-4).sum(axis=1)
    n_ref = np.isclose(p, ref, atol=1e-4).sum(axis=1)
    out = pd.DataFrame({
        "ref": df["ref"].round(0), "open": df["open"].round(0), "close": df["close"].round(0),
        "phiên_chạm_trần": n_ceil, "phiên_chạm_sàn": n_floor, "phiên_bằng_TC": n_ref,
        "đóng_cửa": np.select(
            [np.isclose(df["close"], df["ceil"], atol=1e-4), np.isclose(df["close"], df["floor"], atol=1e-4)],
            ["TRẦN", "SÀN"], default="-"),
    }, index=df.index)
    mask = (out["đóng_cửa"] != "-") | (out["phiên_bằng_TC"] > 0)
    return out[mask]


def daily_range(df: pd.DataFrame) -> pd.Series:
    return df[SESSIONS].max(axis=1) - df[SESSIONS].min(axis=1)


def summary(df: pd.DataFrame) -> dict:
    p = df[SESSIONS].to_numpy().ravel()
    close = df["close"]
    slope = np.polyfit(np.arange(len(close)), close.to_numpy(), 1)[0]
    dr = daily_range(df)
    ms = monthly_stats(df)
    return {
        "mean": p.mean(), "std": p.std(ddof=1), "max": p.max(), "min": p.min(),
        "max_day": df[SESSIONS].max(axis=1).idxmax(), "min_day": df[SESSIONS].min(axis=1).idxmin(),
        "first": df["ref"].iloc[0], "last": close.iloc[-1],
        "change_pct": (close.iloc[-1] / df["ref"].iloc[0] - 1) * 100, "slope": slope,
        "best_day_range": dr.max(), "best_day_range_date": dr.idxmax(),
        "best_day_range_pct": dr.max() / df.loc[dr.idxmax(), "ref"] * 100,
        "wild_month": int(ms["range"].idxmax()), "wild_month_range": ms["range"].max(),
        "touch_days": int(((abnormal_days(df).shape[0]))),
        "all_days_touch": int((np.isclose(df[SESSIONS], df[["ceil"]].to_numpy(), atol=1e-4).any(axis=1)
                               | np.isclose(df[SESSIONS], df[["floor"]].to_numpy(), atol=1e-4).any(axis=1)).sum()),
    }
