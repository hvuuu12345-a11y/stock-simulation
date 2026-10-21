# ABC-11 · Mô phỏng biến động giá cổ phiếu

Video ~60 giây (Full HD, 30 fps) mô phỏng 365 ngày × 7 phiên giao dịch của mã **ABC-11**, kèm trang web
giới thiệu và phân tích (tháng, quý, ngày bất thường, phân bố, xu hướng).
Phần mô phỏng/phân tích chỉ dùng **numpy, pandas, matplotlib**; FastAPI + uvicorn chỉ để phục vụ web.

```
.
├── main.py                 # web FastAPI (uvicorn main:app --reload)
├── requirements.txt
├── render.yaml             # cấu hình deploy lên Render
├── .gitignore
├── data/abc-11.csv         # dữ liệu 365 × 7
├── stock_sim/
│   ├── data.py             # đọc CSV, giá tham chiếu/sàn/trần, quy ước màu
│   ├── analysis.py         # pandas: thống kê tháng, quý, ngày bất thường, xu hướng
│   ├── charts.py           # biểu đồ tĩnh PNG cho web
│   └── video.py            # render video bằng matplotlib + ffmpeg
├── templates/index.html
└── static/
    ├── css/style.css
    ├── img/                # biểu đồ tĩnh (sinh bởi charts.py)
    └── video/clt_simulation.mp4
```

## Chạy local (VS Code)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# cần cài ffmpeg (https://ffmpeg.org) để xuất mp4

python -m stock_sim.video            # render video -> static/video/clt_simulation.mp4 (vài phút)
python -m stock_sim.video --dpi 50 --duration 10   # bản nháp nhanh
python -m stock_sim.video --still 0.5              # xem thử 1 khung hình (still.png)
python -m stock_sim.charts           # sinh lại biểu đồ tĩnh
uvicorn main:app --reload           # mở http://127.0.0.1:8000
```

## Quy ước
- Giá tham chiếu = giá đóng cửa hôm trước (ngày đầu: 10.000, suy ra từ dữ liệu); sàn = 90%, trần = 110%.
- Màu: tham chiếu **vàng**, sàn **xanh dương**, trần **tím**; dưới TC **đỏ**, trên TC **xanh lá**.

## Deploy
1. `git init && git add . && git commit -m "ABC-11 simulation"` rồi push lên GitHub.
2. Render → **New + → Blueprint** → chọn repo (đọc `render.yaml`) → Deploy.
   Video đã render sẵn nằm trong `static/video/` nên Render không cần ffmpeg.
