"""Render video mô phỏng biến động giá cổ phiếu ABC-11 (~60 giây, Full HD).

Chỉ dùng numpy, pandas, matplotlib (matplotlib gọi ffmpeg để mã hoá mp4).

    python -m stock_sim.video                      # render đầy đủ -> static/video/clt_simulation.mp4
    python -m stock_sim.video --still 0.5          # xem nhanh một khung hình (PNG)
    python -m stock_sim.video --dpi 50 --duration 10   # bản nháp nhanh
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox, TransformedBbox

from .analysis import monthly_stats, summary
from .data import BLUE, PALETTE, PURPLE, RED, GREEN, ROOT, SESSIONS, YELLOW, classify, load_prices

BG, PANEL, GRID, TXT, MUTED = "#0a0e1a", "#10162a", "#1f2a44", "#e8edf8", "#8b97b5"
STATUS = ["GIẢM", "TĂNG", "BẰNG THAM CHIẾU", "CHẠM SÀN", "CHẠM TRẦN"]
ARROW = ["▼", "▲", "■", "▼", "▲"]
DEFAULT_OUT = ROOT / "static" / "video" / "clt_simulation.mp4"
FONT = "DejaVu Sans"


def money(v, _=None):
    return f"{v:,.0f}"


class StockVideo:
    def __init__(self, fps=30, duration=60, dpi=100):
        plt.rcParams.update({"font.family": FONT, "text.color": TXT, "axes.edgecolor": GRID})
        self.fps, self.dpi = fps, dpi
        self.n_total = int(fps * duration)
        self.n_intro, self.n_outro = int(fps * 3.5), int(fps * 4.5)
        self.n_anim = self.n_total - self.n_intro - self.n_outro

        df = load_prices()
        self.df, self.sm = df, summary(df)
        self.P = df[SESSIONS].to_numpy()
        self.nd = len(df)                                   # 365 ngày
        self.Y = self.P.ravel()                             # 2555 mức giá
        self.X = (np.arange(self.nd)[:, None] + np.arange(7) / 7).ravel()
        day_of_pt = np.repeat(np.arange(self.nd), 7)
        self.ref, self.floor, self.ceil = (df[c].to_numpy() for c in ("ref", "floor", "ceil"))
        self.codes = classify(self.Y, self.ref[day_of_pt], self.floor[day_of_pt], self.ceil[day_of_pt])
        self.close_codes = classify(self.P[:, -1], self.ref, self.floor, self.ceil)
        self.dates = df.index
        self.pt_month = np.repeat(df.index.month.to_numpy(), 7) - 1          # 0..11
        self.m_start = np.searchsorted(self.pt_month, np.arange(12))
        self.m_end = np.r_[self.m_start[1:], len(self.Y)]
        self.m_final = np.array([self.Y[a:b].mean() for a, b in zip(self.m_start, self.m_end)])
        self.ylim = (0, 13_500)
        self._build()

    # ------------------------------------------------------------------ dựng hình
    def _style(self, ax, title=None):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        if title:
            ax.set_title(title, loc="left", fontsize=11.5, color=MUTED, pad=7)

    def _build(self):
        fig = self.fig = plt.figure(figsize=(19.2, 10.8), dpi=self.dpi, facecolor=BG)
        nd, X, Y = self.nd, self.X, self.Y
        starts = np.r_[0, np.flatnonzero(np.diff(self.dates.month.to_numpy())) + 1]

        # ---- tiêu đề + chú giải màu
        fig.text(0.055, 0.952, "MÔ PHỎNG BIẾN ĐỘNG GIÁ CỔ PHIẾU  ·  ABC-11", fontsize=25, weight="bold")
        fig.text(0.055, 0.920, "365 ngày × 7 phiên = 2.555 mức giá  ·  biên độ ±10% so với giá tham chiếu "
                 "(giá đóng cửa hôm trước)", fontsize=12, color=MUTED)
        x0 = 0.055
        for label, col in [("Tham chiếu", YELLOW), ("Sàn (−10%)", BLUE), ("Trần (+10%)", PURPLE),
                           ("Dưới TC", RED), ("Trên TC", GREEN)]:
            fig.text(x0, 0.886, "■", color=col, fontsize=15, va="center")
            fig.text(x0 + 0.014, 0.886, label, color=TXT, fontsize=11.5, va="center")
            x0 += 0.012 + 0.0075 * len(label) + 0.03
        self.t_day = fig.text(0.97, 0.952, "", fontsize=24, weight="bold", ha="right")
        self.t_date = fig.text(0.97, 0.920, "", fontsize=12.5, color=MUTED, ha="right")

        # ---- (trên) nến phiên theo ngày
        self.ax_c = ax_c = fig.add_axes([0.055, 0.525, 0.64, 0.31])
        self._style(ax_c, "NẾN NGÀY  —  thân: mở cửa → đóng cửa  ·  bấc: thấp nhất → cao nhất trong ngày")
        centre = np.arange(nd) + 3 / 7
        lo, hi = self.P.min(axis=1), self.P.max(axis=1)
        o, c = self.P[:, 0], self.P[:, -1]
        top, bot = np.maximum(o, c), np.minimum(o, c)
        bot = np.minimum(bot, top - 40)
        cc = [to_rgba(PALETTE[k]) for k in self.close_codes]
        bodies = [[(x - .34, b), (x + .34, b), (x + .34, t), (x - .34, t)] for x, b, t in zip(centre, bot, top)]
        self.c_body = PolyCollection(bodies, facecolors=cc, edgecolors=cc, lw=0.4, zorder=3)
        self.c_wick = LineCollection([[(x, l), (x, h)] for x, l, h in zip(centre, lo, hi)],
                                     colors=cc, lw=1.1, zorder=2)
        ax_c.add_collection(self.c_body)
        ax_c.add_collection(self.c_wick)
        self.live_wick, = ax_c.plot([], [], lw=1.4, zorder=4)
        self.live_body = Rectangle((0, 0), .68, 1, zorder=5, ec="white", lw=0.9)
        ax_c.add_patch(self.live_body)
        ax_c.set_xticks(starts)
        ax_c.set_xticklabels([])

        # ---- (dưới) đường giá 7 phiên/ngày + hành lang sàn–trần
        self.ax_l = ax_l = fig.add_axes([0.055, 0.115, 0.64, 0.36], sharex=ax_c)
        self._style(ax_l, "ĐƯỜNG GIÁ THEO PHIÊN  —  màu đoạn thẳng = so với giá tham chiếu của ngày đó")
        for k, s in enumerate(starts):
            if k % 2 == 0:
                ax_l.axvspan(s, starts[k + 1] if k + 1 < len(starts) else nd, color="white", alpha=0.025, lw=0)
        quads = [[(d, f), (d + 1, f), (d + 1, e), (d, e)] for d, (f, e) in enumerate(zip(self.floor, self.ceil))]
        self.l_band = PolyCollection(quads, facecolors=to_rgba("#5b7bff", .075), lw=0, zorder=1)
        ax_l.add_collection(self.l_band)
        self.l_steps = []
        for arr, col in [(self.ref, YELLOW), (self.floor, BLUE), (self.ceil, PURPLE)]:
            lc = LineCollection([[(d, v), (d + 1, v)] for d, v in enumerate(arr)],
                                colors=to_rgba(col, .75), lw=0.9, zorder=2)
            ax_l.add_collection(lc)
            self.l_steps.append(lc)
        pts = np.column_stack([X, Y])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        line_cols = [RED, GREEN, YELLOW, RED, GREEN]
        base = np.array([to_rgba(line_cols[k]) for k in self.codes[1:]])
        self.l_line = []
        for lw, a in [(9, .10), (5, .18), (2.2, 1.0)]:
            col = base.copy()
            col[:, 3] = a
            lc = LineCollection(segs, colors=col, lw=lw, zorder=4, capstyle="round")
            ax_l.add_collection(lc)
            self.l_line.append(lc)
        tc = np.flatnonzero(np.isin(self.codes, (3, 4)))       # phiên chạm sàn / trần
        self.l_marks = ax_l.scatter(X[tc], Y[tc], s=4.5, zorder=5, linewidths=0,
                                    c=[PALETTE[k] for k in self.codes[tc]], alpha=.95)
        self.head_glow = ax_l.scatter([], [], s=420, zorder=6, alpha=.28, linewidths=0)
        self.head = ax_l.scatter([], [], s=70, zorder=7, edgecolors="white", linewidths=1.4)
        self.head_txt = ax_l.text(0, 0, "", fontsize=13, weight="bold", zorder=8)
        self.max_line = ax_l.axhline(0, color=GREEN, lw=0.9, ls=(0, (1, 3)), zorder=3)
        self.min_line = ax_l.axhline(0, color=RED, lw=0.9, ls=(0, (1, 3)), zorder=3)
        self.max_txt = ax_l.text(366.5, 0, "", color=GREEN, fontsize=11.5, va="center")
        self.min_txt = ax_l.text(366.5, 0, "", color=RED, fontsize=11.5, va="center")
        ax_l.set_xlim(-4, 400)
        ax_l.set_ylim(*self.ylim)
        ax_l.yaxis.set_major_formatter(FuncFormatter(money))
        ax_c.yaxis.set_major_formatter(FuncFormatter(money))
        ax_l.set_xticks(starts)
        ax_l.set_xticklabels([f"T{m}" for m in range(1, 13)])
        ax_l.set_ylabel("VND", color=MUTED)
        ax_c.set_ylabel("VND", color=MUTED)
        ax_c.set_ylim(*self.ylim)
        ax_c.tick_params(labelbottom=False)
        fig.text(0.055, 0.062, "Vì sao đường đi xuống mà vẫn xanh?  Vì màu được so với GIÁ THAM CHIẾU của ngày, "
                 "không phải so với điểm liền trước.", fontsize=11.5, color=MUTED, style="italic")
        self._milestones()

        # ---- thanh tiến độ
        self.ax_p = ax_p = fig.add_axes([0.055, 0.028, 0.64, 0.009])
        ax_p.set_xlim(0, 1)
        ax_p.set_axis_off()
        ax_p.add_patch(Rectangle((0, 0), 1, 1, color=GRID))
        self.prog = Rectangle((0, 0), 0, 1, color=GREEN)
        ax_p.add_patch(self.prog)

        self._dashboard()
        self._overlay()
        self.clipped_c = [self.c_body, self.c_wick]
        self.clipped_l = [self.l_band, *self.l_steps, *self.l_line, self.l_marks]

    def _milestones(self):
        ax, sm, df = self.ax_l, self.sm, self.df
        imax = int(np.argmax(self.Y))
        imin = int(np.argmin(self.Y))
        spec = [
            (0.0, 10_000, "Mở đầu: giá TC = 10,000", (60, 28), "left"),
            (self.X[imin], self.Y[imin], f"ĐÁY NĂM  {sm['min']:,.0f}  ({sm['min_day']:%d/%m})", (-20, -75), "right"),
            (self.X[imax], self.Y[imax], f"ĐỈNH NĂM  {sm['max']:,.0f}  ({sm['max_day']:%d/%m})", (-90, -60), "right"),
        ]
        self.miles = []
        for x, y, text, off, ha in spec:
            a = ax.annotate(text, (x, y), xytext=off, textcoords="offset points", ha=ha, fontsize=12,
                            weight="bold", color=TXT, zorder=9,
                            bbox=dict(boxstyle="round,pad=0.35", fc="#1b2442", ec=GRID),
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
            a.set_visible(False)
            self.miles.append((x, a))

    def _dashboard(self):
        fig = self.fig
        # bảng giá trực tiếp
        self.ax_t = ax_t = fig.add_axes([0.725, 0.575, 0.245, 0.26])
        ax_t.set_facecolor(PANEL)
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        for s in ax_t.spines.values():
            s.set_color(GRID)
        tr = ax_t.transAxes
        ax_t.text(0.05, 0.89, "GIÁ TRỰC TIẾP  ·  VND", transform=tr, fontsize=11, color=MUTED)
        self.t_price = ax_t.text(0.05, 0.57, "", transform=tr, fontsize=45, weight="bold")
        self.t_chg = ax_t.text(0.05, 0.43, "", transform=tr, fontsize=13, weight="bold")
        self.t_ref = ax_t.text(0.05, 0.31, "", transform=tr, fontsize=11, color=MUTED)
        self.t_stat = ax_t.text(0.95, 0.89, "", transform=tr, fontsize=11.5, weight="bold", ha="right")
        self.ax_g = ax_g = fig.add_axes([0.735, 0.615, 0.225, 0.018])
        grad = LinearSegmentedColormap.from_list("g", [BLUE, RED, YELLOW, GREEN, PURPLE])
        ax_g.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=grad, extent=(0, 1, 0, 1), alpha=.9)
        ax_g.set_xticks([])
        ax_g.set_yticks([])
        for s in ax_g.spines.values():
            s.set_visible(False)
        self.g_mark, = ax_g.plot([.5, .5], [-.5, 1.5], color="white", lw=3, clip_on=False)
        for x, t in [(0, "SÀN −10%"), (.5, "TC"), (1, "TRẦN +10%")]:
            ax_g.text(x, -1.1, t, ha=["left", "center", "right"][int(x * 2)], fontsize=9.5, color=MUTED,
                      transform=ax_g.transData, va="top")

        # cột: giá trung bình theo tháng (đang tích luỹ)
        self.ax_m = ax_m = fig.add_axes([0.725, 0.315, 0.245, 0.195])
        self._style(ax_m, "GIÁ TRUNG BÌNH THEO THÁNG (tích luỹ)")
        self.bars = ax_m.bar(np.arange(12), np.zeros(12), width=.72, zorder=3)
        self.bar_txt = [ax_m.text(i, 0, "", ha="center", va="bottom", fontsize=8.2, color=TXT) for i in range(12)]
        ax_m.set_ylim(0, self.m_final.max() * 1.22)
        ax_m.set_xticks(range(12))
        ax_m.set_xticklabels([str(i) for i in range(1, 13)])
        ax_m.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))

        # histogram phân bố giá tích luỹ
        self.ax_h = ax_h = fig.add_axes([0.725, 0.075, 0.245, 0.185])
        self._style(ax_h, "PHÂN BỐ GIÁ TÍCH LUỸ  (vạch trắng = giá hiện tại)")
        self.edges = np.linspace(0, 13_000, 27)
        mid = (self.edges[:-1] + self.edges[1:]) / 2
        cmap = LinearSegmentedColormap.from_list("h", [RED, YELLOW, GREEN])
        self.hist = ax_h.bar(mid, np.zeros(len(mid)), width=np.diff(self.edges) * .88, zorder=3,
                             color=cmap(mid / 13_000))
        ax_h.set_ylim(0, np.histogram(self.Y, self.edges)[0].max() * 1.1)
        ax_h.set_xlim(0, 13_000)
        ax_h.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
        self.h_line = ax_h.axvline(0, color="white", lw=2, zorder=5)

    def _overlay(self):
        fig, sm = self.fig, self.sm
        self.ax_o = ax_o = fig.add_axes([0, 0, 1, 1], zorder=20)
        ax_o.set_axis_off()
        ax_o.set_xlim(0, 1)
        ax_o.set_ylim(0, 1)
        self.o_bg = Rectangle((0, 0), 1, 1, color=BG, zorder=0)
        ax_o.add_patch(self.o_bg)
        # --- intro
        self.intro = [
            ax_o.text(.5, .60, "MÔ PHỎNG BIẾN ĐỘNG\nGIÁ CỔ PHIẾU", ha="center", va="center", fontsize=56,
                      weight="bold", linespacing=1.15),
            ax_o.text(.5, .40, "Mã  ABC-11   ·   365 ngày  ×  7 phiên giao dịch", ha="center", fontsize=22,
                      color=YELLOW),
            ax_o.text(.5, .31, "Sàn −10%   ·   Tham chiếu   ·   Trần +10%", ha="center", fontsize=17, color=MUTED),
            ax_o.text(.5, .12, "numpy  ·  pandas  ·  matplotlib", ha="center", fontsize=15, color=MUTED),
        ]
        # --- outro
        ms = monthly_stats(self.df)
        hi_m, lo_m = int(ms["mean"].idxmax()), int(ms["mean"].idxmin())
        trend = "đi xuống nhẹ" if sm["slope"] < -1 else ("đi lên nhẹ" if sm["slope"] > 1 else "đi ngang")
        rows = [
            ("Giá đầu năm → cuối năm", f"{sm['first']:,.0f} → {sm['last']:,.0f}  ({sm['change_pct']:+.1f}%)"),
            ("Trung bình ± độ lệch chuẩn", f"{sm['mean']:,.0f} ± {sm['std']:,.0f}"),
            ("Đỉnh năm", f"{sm['max']:,.0f}   ({sm['max_day']:%d/%m})"),
            ("Đáy năm", f"{sm['min']:,.0f}   ({sm['min_day']:%d/%m})"),
            ("Biên độ trong ngày lớn nhất", f"{sm['best_day_range']:,.0f} (≈{sm['best_day_range_pct']:.0f}% TC)   "
                                           f"({sm['best_day_range_date']:%d/%m})"),
            ("Tháng dao động mạnh nhất", f"Tháng {sm['wild_month']}  (biên độ {sm['wild_month_range']:,.0f})"),
            ("Giá TB tháng cao nhất / thấp nhất", f"Tháng {hi_m} ({ms['mean'][hi_m]:,.0f})  /  "
                                                 f"Tháng {lo_m} ({ms['mean'][lo_m]:,.0f})"),
            ("Xu hướng cả năm (hồi quy)", f"{sm['slope']:+.1f} VND/ngày  →  {trend}, dạng chữ U"),
        ]
        self.outro = [ax_o.text(.5, .86, "TỔNG KẾT NĂM", ha="center", fontsize=38, weight="bold")]
        for i, (k, v) in enumerate(rows):
            y = .75 - i * .075
            self.outro.append(ax_o.text(.17, y, k, fontsize=18, color=MUTED, va="center"))
            self.outro.append(ax_o.text(.83, y, v, fontsize=19, weight="bold", color=TXT, va="center", ha="right"))
        self.outro.append(ax_o.text(.5, .12, "Đường đi xuống nhưng vẫn xanh: màu so với GIÁ THAM CHIẾU của ngày,\n"
                                    "không so với điểm liền trước.", ha="center", fontsize=16, color=YELLOW,
                                    linespacing=1.5))

    # ------------------------------------------------------------------ cập nhật khung
    def _clip(self, arts, ax, x1):
        box = TransformedBbox(Bbox([[-10, -1e5], [x1, 1e5]]), ax.transData)
        for a in arts:
            a.set_clip_box(box)

    def draw_frame(self, i):
        if i < self.n_intro:                                     # --- intro
            p, a_in, a_out = 0.0, 1.0 if i < .55 * self.n_intro else 1 - (i - .55 * self.n_intro) / (.45 * self.n_intro), 0
        elif i < self.n_intro + self.n_anim:                     # --- chạy dữ liệu
            p, a_in, a_out = (i - self.n_intro) / (self.n_anim - 1) * self.nd, 0, 0
        else:                                                    # --- tổng kết
            p, a_in = float(self.nd), 0
            a_out = min(1.0, (i - self.n_intro - self.n_anim) / (0.8 * self.fps))
        self._update(p, i)
        self.o_bg.set_alpha(max(a_in, .90 * a_out))
        for t in self.intro:
            t.set_alpha(a_in)
            t.set_visible(a_in > 0)
        for t in self.outro:
            t.set_alpha(a_out)
            t.set_visible(a_out > 0)
        self.o_bg.set_visible(a_in > 0 or a_out > 0)

    def _update(self, p, i):
        X, Y, nd = self.X, self.Y, self.nd
        d = min(int(p), nd - 1)
        hx = min(p, X[-1])
        hy = float(np.interp(hx, X, Y))
        n = int(np.searchsorted(X, hx, side="right"))
        si = min(int((p - d) * 7), 6)                           # phiên hiện tại (0..6)
        price = self.P[d, si]
        ref, flo, cei = self.ref[d], self.floor[d], self.ceil[d]
        code = int(classify(price, ref, flo, cei))
        col = PALETTE[code]

        # clip các lớp đã vẽ sẵn
        done = int(p) if p < nd else nd
        self._clip(self.clipped_c, self.ax_c, done)
        self._clip(self.clipped_l, self.ax_l, hx)

        # nến của ngày đang chạy
        live = done < nd
        self.live_wick.set_visible(live)
        self.live_body.set_visible(live)
        if live:
            cur = self.P[d, :si + 1]
            op = self.P[d, 0]
            xc = d + 3 / 7
            ccode = int(classify(price, ref, flo, cei))
            self.live_wick.set_data([xc, xc], [cur.min(), cur.max()])
            self.live_wick.set_color(PALETTE[ccode])
            self.live_body.set_xy((xc - .34, min(op, price)))
            self.live_body.set_height(max(abs(price - op), 40))
            self.live_body.set_facecolor(PALETTE[ccode])

        # đầu vẽ + nhãn
        pulse = 1 + .25 * np.sin(i * .35)
        self.head.set_offsets([[hx, hy]])
        self.head.set_facecolor(col)
        self.head_glow.set_offsets([[hx, hy]])
        self.head_glow.set_facecolor(col)
        self.head_glow.set_sizes([420 * pulse])
        self.head_txt.set_position((hx - 6 if hx > 330 else hx + 5, hy + (330 if hx > 330 else 0)))
        self.head_txt.set_ha("right" if hx > 330 else "left")
        self.head_txt.set_text(f"{hy:,.0f}")
        self.head_txt.set_color(col)

        # đường đỉnh / đáy chạy
        hi, lo = max(Y[:n].max(), hy), min(Y[:n].min(), hy)
        self.max_line.set_ydata([hi, hi])
        self.min_line.set_ydata([lo, lo])
        self.max_txt.set_position((366.5, hi))
        self.min_txt.set_position((366.5, lo))
        self.max_txt.set_text(f"{hi:,.0f}")
        self.min_txt.set_text(f"{lo:,.0f}")
        for x, a in self.miles:
            a.set_visible(hx >= x + 0.01)

        # bảng giá trực tiếp
        diff = price - ref
        self.t_price.set_text(f"{price:,.0f}")
        self.t_price.set_color(col)
        self.t_chg.set_text(f"{ARROW[code]} {diff:+,.0f}   ({diff / ref * 100:+.2f}%)")
        self.t_chg.set_color(col)
        self.t_ref.set_text(f"TC {ref:,.0f}   ·   sàn {flo:,.0f}   ·   trần {cei:,.0f}")
        self.t_stat.set_text(STATUS[code])
        self.t_stat.set_color(col)
        self.g_mark.set_xdata([(price - flo) / (cei - flo)] * 2)
        self.t_day.set_text(f"NGÀY {d + 1}/{nd}")
        self.t_date.set_text(f"{self.dates[d]:%d/%m/%Y}   ·   phiên {si + 1}/7")
        self.prog.set_width(min(p / nd, 1))
        self.prog.set_color(col)

        # cột trung bình tháng
        cm = int(self.pt_month[min(n, len(Y) - 1) - 1]) if n > 0 else 0
        means = np.zeros(12)
        means[:cm] = self.m_final[:cm]
        means[cm] = Y[self.m_start[cm]:max(n, self.m_start[cm] + 1)].mean()
        for m in range(12):
            h = means[m]
            self.bars[m].set_height(h)
            up = m == 0 or h >= self.m_final[m - 1]
            self.bars[m].set_facecolor(GREEN if up else RED)
            self.bars[m].set_alpha(1.0 if m < cm else .65)
            self.bar_txt[m].set_position((m, h))
            self.bar_txt[m].set_text(f"{h / 1000:.1f}k" if h > 0 else "")

        # histogram tích luỹ
        counts, _ = np.histogram(Y[:max(n, 1)], self.edges)
        for r, h in zip(self.hist, counts):
            r.set_height(h)
        self.h_line.set_xdata([hy, hy])

    # ------------------------------------------------------------------ xuất video
    def render(self, out=DEFAULT_OUT):
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.FFMpegWriter(
            fps=self.fps, codec="libx264", metadata={"title": "Mo phong bien dong gia ABC-11"},
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart"])
        t0 = time.time()
        with writer.saving(self.fig, str(out), self.dpi):
            for i in range(self.n_total):
                self.draw_frame(i)
                writer.grab_frame()
                if i % 60 == 0:
                    el = time.time() - t0
                    eta = el / (i + 1) * (self.n_total - i - 1)
                    print(f"  khung {i:4d}/{self.n_total}   đã chạy {el:5.0f}s   còn ~{eta:5.0f}s", flush=True)
        print(f"Xong: {out}  ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")

    def still(self, frac, out):
        self.draw_frame(int(frac * (self.n_total - 1)))
        self.fig.savefig(out, dpi=self.dpi, facecolor=BG)
        print("Đã lưu", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=60, help="thời lượng (giây)")
    ap.add_argument("--dpi", type=int, default=100, help="100 = 1920x1080, 50 = 960x540")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--still", type=float, default=None, help="chỉ lưu 1 khung (0..1) ra PNG")
    a = ap.parse_args()
    sv = StockVideo(a.fps, a.duration, a.dpi)
    if a.still is not None:
        sv.still(a.still, a.out if a.out.endswith(".png") else "still.png")
    else:
        sv.render(a.out)


if __name__ == "__main__":
    main()
