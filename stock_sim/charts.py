"""Biểu đồ tĩnh (PNG) cho phần phân tích trên trang web: câu 1, 4, 5."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import long_prices, monthly_stats
from .data import GREEN, RED, ROOT, SESSIONS, YELLOW, load_prices

IMG = ROOT / "static" / "img"
BG, PANEL, GRID, TXT, MUTED = "#0a0e1a", "#10162a", "#1f2a44", "#e8edf8", "#8b97b5"


def _ax(figsize=(9, 4.6)):
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TXT})
    fig, ax = plt.subplots(figsize=figsize, dpi=130, facecolor=BG)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, name):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(IMG / name, facecolor=BG)
    plt.close(fig)


def monthly_bar(df):
    ms = monthly_stats(df)
    fig, ax = _ax()
    cols = [GREEN if i == 0 or ms["mean"].iloc[i] >= ms["mean"].iloc[i - 1] else RED for i in range(12)]
    ax.bar(ms.index, ms["mean"], yerr=ms["std"], color=cols, capsize=3, ecolor=MUTED, zorder=3)
    ax.set(xlabel="Tháng", ylabel="Giá trung bình (VND)", xticks=range(1, 13))
    ax.set_title("Giá trung bình mỗi tháng (thanh sai số = độ lệch chuẩn)", color=TXT)
    _save(fig, "monthly_mean.png")


def monthly_box(df):
    lp = long_prices(df)
    fig, ax = _ax()
    data = [lp.loc[lp.month == m, "price"] for m in range(1, 13)]
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color=YELLOW, lw=2),
                    flierprops=dict(marker="o", ms=3, mfc=RED, mec=RED), whiskerprops=dict(color=MUTED),
                    capprops=dict(color=MUTED))
    for b in bp["boxes"]:
        b.set(facecolor="#2a3a6e", edgecolor=MUTED)
    ax.set(xlabel="Tháng", ylabel="Giá (VND)")
    ax.set_title("Phân bố giá theo tháng (boxplot)", color=TXT)
    _save(fig, "monthly_box.png")


def half_year_hist(df):
    lp = long_prices(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=130, facecolor=BG, sharey=True)
    bins = np.linspace(0, 13_000, 27)
    for ax, (qs, title, col) in zip(axes, [((1, 2), "Quý 1 + Quý 2", GREEN), ((3, 4), "Quý 3 + Quý 4", RED)]):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color=GRID, lw=.6)
        sub = lp[lp.quarter.isin(qs)]["price"]
        ax.hist(sub, bins=bins, color=col, alpha=.85, zorder=3)
        ax.axvline(sub.mean(), color=YELLOW, ls="--", lw=1.5, label=f"TB = {sub.mean():,.0f}")
        ax.set_title(title, color=TXT)
        ax.set_xlabel("Giá (VND)", color=MUTED)
        ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT)
    axes[0].set_ylabel("Số mức giá", color=MUTED)
    _save(fig, "hist_halves.png")


def trend(df):
    fig, ax = _ax((10, 4.4))
    x = np.arange(len(df))
    ax.plot(x, df["close"], color=GREEN, lw=1.2, label="Giá đóng cửa")
    ax.plot(x, df["close"].rolling(30, min_periods=1).mean(), color=YELLOW, lw=2, label="Trung bình trượt 30 ngày")
    a, b = np.polyfit(x, df["close"], 1)
    ax.plot(x, a * x + b, color=RED, ls="--", lw=1.6, label=f"Hồi quy tuyến tính ({a:+.1f} VND/ngày)")
    ax.set(xlabel="Ngày", ylabel="VND")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TXT)
    ax.set_title("Xu hướng giá đóng cửa trong năm", color=TXT)
    _save(fig, "trend.png")


def build_all():
    df = load_prices()
    for f in (monthly_bar, monthly_box, half_year_hist, trend):
        f(df)
    print("Đã tạo biểu đồ trong", IMG)


if __name__ == "__main__":
    build_all()
