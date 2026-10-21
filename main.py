"""Ứng dụng web (FastAPI + uvicorn) giới thiệu video mô phỏng + kết quả phân tích cổ phiếu ABC-11.

Chạy local :  uvicorn main:app --reload      ->  http://127.0.0.1:8000   (hoặc: python main.py)
Chạy Render:  uvicorn main:app --host 0.0.0.0 --port $PORT   (xem render.yaml)

Lưu ý: phần mô phỏng/phân tích (thư mục stock_sim/) chỉ dùng numpy, pandas, matplotlib.
FastAPI/uvicorn chỉ dùng để phục vụ trang web.
"""
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from stock_sim.analysis import abnormal_days, monthly_stats, quarterly_stats, summary
from stock_sim.data import load_prices

BASE = Path(__file__).resolve().parent
app = FastAPI(title="ABC-11 stock simulation")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

df = load_prices()
SM = summary(df)


def _table(frame, **kw):
    return frame.to_html(classes="tbl", border=0, **kw)


def build_context():
    ms, qs, ab = monthly_stats(df), quarterly_stats(df), abnormal_days(df)
    ms_v = ms.rename(columns={"mean": "TB", "std": "Độ lệch chuẩn", "min": "Thấp nhất",
                              "max": "Cao nhất", "range": "Biên độ"})
    ms_v.index.name = "Tháng"
    qs_v = qs.rename(columns={"mean": "TB", "std": "Độ lệch chuẩn", "min": "Thấp nhất",
                              "max": "Cao nhất", "range": "Biên độ", "cv_%": "CV (%)"})
    qs_v.index.name = "Quý"
    ab_v = ab.copy()
    ab_v.index = ab_v.index.strftime("%d/%m/%Y")
    ab_v.index.name = "Ngày"
    return {
        "sm": SM,
        "monthly_table": _table(ms_v, float_format=lambda v: f"{v:,.0f}"),
        "quarter_table": _table(qs_v, float_format=lambda v: f"{v:,.1f}"),
        "abnormal_table": _table(ab_v),
        "n_abnormal": len(ab),
        "n_close_ceil": int((ab["đóng_cửa"] == "TRẦN").sum()),
        "n_close_floor": int((ab["đóng_cửa"] == "SÀN").sum()),
        "hi_month": int(ms["mean"].idxmax()), "lo_month": int(ms["mean"].idxmin()),
        "ms": ms, "qs": qs,
    }


CONTEXT = build_context()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", CONTEXT)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "ok"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
