"""Đọc dữ liệu và tính giá tham chiếu / sàn / trần."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "abc-11.csv"
SESSIONS = ["open", "s1", "s2", "s3", "s4", "s5", "close"]
FIRST_REF = 10_000.0          # giá tham chiếu của ngày đầu tiên (suy ra từ dữ liệu)
START_DATE = "2025-01-01"

# Màu quy ước (nền tối)
YELLOW, BLUE, PURPLE = "#ffd400", "#3b82f6", "#b45cff"
RED, GREEN = "#ff3b4e", "#22d36b"


def load_prices(path=DATA_FILE) -> pd.DataFrame:
    """Trả về DataFrame 365 dòng: 7 phiên + ref, floor, ceil (index = ngày)."""
    raw = pd.read_csv(path, header=None, names=SESSIONS)
    raw.index = pd.date_range(START_DATE, periods=len(raw), name="date")
    raw["ref"] = np.r_[FIRST_REF, raw["close"].to_numpy()[:-1]]
    raw["floor"] = raw["ref"] * 0.9
    raw["ceil"] = raw["ref"] * 1.1
    return raw


def classify(price, ref, floor, ceil):
    """0=đỏ(giảm) 1=xanh(tăng) 2=vàng(=TC) 3=xanh dương(sàn) 4=tím(trần)."""
    price, ref = np.asarray(price, float), np.asarray(ref, float)
    floor, ceil = np.asarray(floor, float), np.asarray(ceil, float)
    code = np.where(price > ref, 1, 0)
    code = np.where(np.isclose(price, ref, atol=1e-4), 2, code)
    code = np.where(np.isclose(price, floor, atol=1e-4), 3, code)
    code = np.where(np.isclose(price, ceil, atol=1e-4), 4, code)
    return code


PALETTE = [RED, GREEN, YELLOW, BLUE, PURPLE]
