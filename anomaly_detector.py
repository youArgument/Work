import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import glob
from scipy import stats

st.set_page_config(layout="wide", page_title="Anomaly Detector")

st.title("🔍 Anomaly Detector — Поиск аномалий на магнитограмме")

base_dir = "ROW_DATA/df"

stands = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
sensors = ["mdr", "umdp"]

left_col, right_col = st.columns([1, 3])

with left_col:
    st.header("⚙️ Данные")
    selected_stand = st.selectbox("Стенд", stands, key="an_stand")
    selected_sensor = st.selectbox("Датчик", sensors, key="an_sensor")

    npz_path = os.path.join(base_dir, selected_stand, selected_sensor, "npz_datasets")
    files = sorted(glob.glob(os.path.join(npz_path, "*.npz")))

    if not files:
        st.error("Файлы не найдены")
        st.stop()

    show_all = st.checkbox("Все секторы", value=False, key="an_all")
    selected_file_idx = st.selectbox("Сектор", range(len(files)), format_func=lambda i: os.path.basename(files[i]), key="an_file")

    if show_all:
        all_mag, all_vel, all_acc = [], [], []
        for f in files:
            d = np.load(f)
            all_mag.append(d["magnetogram"])
            all_vel.append(d["velocity"])
            all_acc.append(d["accelerometer"])
        magnetogram = np.concatenate(all_mag, axis=0)
        velocity = np.concatenate(all_vel, axis=0)
        accelerometer = np.concatenate(all_acc, axis=0)
    else:
        d = np.load(files[selected_file_idx])
        magnetogram = d["magnetogram"]
        velocity = d["velocity"]
        accelerometer = d["accelerometer"]

    n_steps, n_channels = magnetogram.shape

    st.divider()
    st.header("🔬 Метод детекции")

    method = st.radio(
        "Алгоритм",
        ["Z-Score", "MAD (робастный)", "Скользящее окно", "Градиент"],
        index=0,
        key="an_method",
        horizontal=True,
    )

    st.divider()
    st.header("📏 Пороги")

    if method == "Z-Score":
        threshold = st.slider("Порог σ", 1.0, 8.0, 3.0, 0.1, key="an_thresh")
    elif method == "MAD (робастный)":
        threshold = st.slider("MAD-множитель", 1.0, 15.0, 3.5, 0.1, key="an_thresh")
    elif method == "Скользящее окно":
        win_size = st.slider("Размер окна", 10, min(500, n_steps), min(100, n_steps), key="an_win")
        threshold = st.slider("Отклонение от медианы", 10, int(magnetogram.max()), int(magnetogram.std()), 1, key="an_thresh")
    else:
        threshold = st.slider("Порог градиента", 1.0, 20.0, 3.0, 0.1, key="an_thresh")

    scope = st.radio("Область", ["По каналам", "Глобальный", "По блокам"], index=0, key="an_scope", horizontal=True)

    st.divider()
    st.header("🎨 Отображение")

    cmap_choice = st.selectbox("Colormap", ["Viridis", "Plasma", "Hot", "coolwarm", "Turbo", "Inferno", "RdBu_r"], index=0, key="an_cmap")
    highlight_color = st.selectbox("Цвет аномалий", ["red", "yellow", "cyan", "magenta", "lime"], index=0, key="an_color")

st.caption(f"Шагов: {n_steps} | Каналов: {n_channels}")

def detect_zscore(data, thresh, scope):
    if scope == "По каналам":
        mu = data.mean(axis=0)
        sigma = data.std(axis=0)
        sigma[sigma == 0] = 1
        residual = np.abs(data - mu) / sigma
    elif scope == "По блокам":
        block_h, block_w = max(1, n_steps // 8), max(1, n_channels // 8)
        mu = np.zeros_like(data)
        sigma = np.ones_like(data)
        for i in range(0, n_steps, block_h):
            for j in range(0, n_channels, block_w):
                block = data[i:min(i+block_h, n_steps), j:min(j+block_w, n_channels)]
                m = block.mean()
                s = block.std()
                if s == 0:
                    s = 1
                mu[i:min(i+block_h, n_steps), j:min(j+block_w, n_channels)] = m
                sigma[i:min(i+block_h, n_steps), j:min(j+block_w, n_channels)] = s
        residual = np.abs(data - mu) / sigma
    else:
        mu = data.mean()
        sigma = data.std()
        if sigma == 0:
            sigma = 1
        residual = np.abs(data - mu) / sigma
    mask = residual > thresh
    return mask, residual

def detect_mad(data, thresh, scope):
    if scope == "По каналам":
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        mad[mad == 0] = 1
        residual = np.abs(data - median) / (mad * 1.4826)
    elif scope == "По блокам":
        block_h, block_w = max(1, n_steps // 8), max(1, n_channels // 8)
        median = np.zeros_like(data, dtype=float)
        mad = np.ones_like(data, dtype=float)
        for i in range(0, n_steps, block_h):
            for j in range(0, n_channels, block_w):
                block = data[i:min(i+block_h, n_steps), j:min(j+block_w, n_channels)]
                m = np.median(block)
                md = np.median(np.abs(block - m))
                if md == 0:
                    md = 1
                median[i:min(i+block_h, n_steps), j:min(j+block_w, n_channels)] = m
                mad[i:min(i+block_h, n_steps), j:min(j+block_w, n_channels)] = md
        residual = np.abs(data - median) / (mad * 1.4826)
    else:
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            mad = 1
        residual = np.abs(data - median) / (mad * 1.4826)
    mask = residual > thresh
    return mask, residual

def detect_sliding_window(data, thresh, scope):
    win = int(win_size)
    if scope == "По каналам":
        median_bg = np.zeros_like(data, dtype=float)
        for ch in range(n_channels):
            col = data[:, ch]
            for i in range(n_steps):
                lo = max(0, i - win // 2)
                hi = min(n_steps, i + win // 2)
                median_bg[i, ch] = np.median(col[lo:hi])
        residual = np.abs(data - median_bg)
    else:
        median_bg = np.zeros_like(data, dtype=float)
        for i in range(n_steps):
            lo = max(0, i - win // 2)
            hi = min(n_steps, i + win // 2)
            median_bg[i, :] = np.median(data[lo:hi], axis=0)
        residual = np.abs(data - median_bg)
    mask = residual > thresh
    return mask, residual

def detect_gradient(data, thresh, scope):
    grad = np.abs(np.gradient(data, axis=0))
    if scope == "По каналам":
        mu = grad.mean(axis=0)
        sigma = grad.std(axis=0)
        sigma[sigma == 0] = 1
        residual = (grad - mu) / sigma
    else:
        mu = grad.mean()
        sigma = grad.std()
        if sigma == 0:
            sigma = 1
        residual = (grad - mu) / sigma
    mask = residual > thresh
    return mask, residual

with left_col:
    st.divider()
    st.subheader("Вычисление...")

if method == "Z-Score":
    anomaly_mask, scores = detect_zscore(magnetogram, threshold, scope)
elif method == "MAD (робастный)":
    anomaly_mask, scores = detect_mad(magnetogram, threshold, scope)
elif method == "Скользящее окно":
    anomaly_mask, scores = detect_sliding_window(magnetogram, threshold, scope)
else:
    anomaly_mask, scores = detect_gradient(magnetogram, threshold, scope)

n_anomalies = int(anomaly_mask.sum())
anomaly_pct = 100.0 * n_anomalies / (n_steps * n_channels)

with left_col:
    st.success("Готово!")
    st.metric("Аномалий (пикселей)", f"{n_anomalies:,}")
    st.metric("Доля данных", f"{anomaly_pct:.2f}%")

    anom_steps = np.where(anomaly_mask.any(axis=1))[0]
    if len(anom_steps) > 0:
        st.metric("Шагов с аномалиями", f"{len(anom_steps)}")
        st.metric("Макс. шаг", f"{anom_steps.max()}")

    anom_channels = np.where(anomaly_mask.any(axis=0))[0]
    st.metric("Каналов с аномалиями", f"{len(anom_channels)}")

    st.divider()
    st.subheader("Топ-аномалии")
    flat_scores = scores.flatten()
    flat_mask = anomaly_mask.flatten()
    anom_indices = np.where(flat_mask)[0]
    anom_scores = flat_scores[anom_indices]
    top_k = min(10, len(anom_scores))
    top_local = anom_indices[np.argsort(-anom_scores)[:top_k]]
    top_steps = top_local // n_channels
    top_chs = top_local % n_channels
    for i, (s, ch) in enumerate(zip(top_steps, top_chs)):
        st.caption(f"#{i+1}: шаг {s}, канал #{ch}, score={scores[s, ch]:.1f}, val={magnetogram[s, ch]}")

with right_col:
    st.header("🗺️ Карта аномалий")

    display = magnetogram.astype(float)
    display[anomaly_mask] = np.nan

    fig_heat = go.Figure()

    fig_heat.add_trace(go.Heatmap(
        z=magnetogram.T,
        colorscale=cmap_choice,
        colorbar=dict(title="Signal", x=1.02),
        hovertemplate="Step: %{x}<br>Ch: %{y}<br>Val: %{z}<extra></extra>",
        showscale=True,
    ))

    anom_y = np.where(anomaly_mask.any(axis=1))[0]
    for ay in anom_y[:500]:
        fig_heat.add_hline(
            y=ay, line=dict(color=highlight_color, width=1.5, dash="dot"),
            opacity=0.5,
        )

    fig_heat.update_layout(
        title=f"Магнитограмма — аномалии ({anomaly_pct:.2f}% данных, {n_anomalies:,} пикселей)",
        xaxis_title="Step (глубина)",
        yaxis_title="Channel",
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()
    st.header("📊 Карта значимости (scores)")

    fig_scores = go.Figure(data=go.Heatmap(
        z=scores.T,
        colorscale="Hot",
        colorbar=dict(title="Score", x=1.02),
        hovertemplate="Step: %{x}<br>Ch: %{y}<br>Score: %{z:.1f}<extra></extra>",
    ))
    fig_scores.update_layout(
        title=f"Значимость ({method})",
        xaxis_title="Step",
        yaxis_title="Channel",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_scores, use_container_width=True)

    st.divider()
    st.header("📈 Распределение scores")

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=flat_scores[~flat_mask], nbinsx=100, name="Норма", opacity=0.7, marker_color="steelblue"))
    if n_anomalies > 0:
        fig_hist.add_trace(go.Histogram(x=flat_scores[flat_mask], nbinsx=100, name="Аномалия", opacity=0.7, marker_color="coral"))
    fig_hist.update_layout(
        barmode="overlay",
        xaxis_title="Score",
        yaxis_title="Count",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.95, y=0.95),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()
    st.header("🎯 Срез по каналам")

    ch_start = st.slider("Канал от", 0, n_channels - 1, 0, key="an_chs")
    ch_end = st.slider("Канал до", 1, n_channels, min(64, n_channels), key="an_che")
    ch_end = max(ch_end, ch_start + 1)

    fig_ch = go.Figure()
    for ch in range(ch_start, min(ch_end, ch_start + 128)):
        is_anom = anomaly_mask[:, ch]
        fig_ch.add_trace(go.Scatter(
            x=np.arange(n_steps), y=magnetogram[:, ch],
            mode="lines", name=f"Ch {ch}",
            line=dict(width=0.7, color="steelblue"),
            opacity=0.3,
            showlegend=False,
        ))
        if is_anom.any():
            fig_ch.add_trace(go.Scatter(
                x=np.where(is_anom)[0], y=magnetogram[is_anom, ch],
                mode="markers", name=f"Ch {ch} anom",
                marker=dict(color=highlight_color, size=3),
                showlegend=False,
            ))

    fig_ch.update_layout(
        title=f"Каналы {ch_start}–{min(ch_end-1, ch_start+127)}",
        xaxis_title="Step",
        yaxis_title="Value",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ch, use_container_width=True)

    st.divider()
    st.header("🔎 Поперечное сечение в точке аномалии")

    if len(anom_steps) > 0:
        inspect_step = st.slider("Шаг для инспекции", int(anom_steps.min()), int(anom_steps.max()), int(anom_steps[0]), key="an_inspect")
    else:
        inspect_step = 0

    cross = magnetogram[inspect_step, :]
    cross_anom = anomaly_mask[inspect_step, :]
    cross_scores = scores[inspect_step, :]
    angles = np.linspace(0, 360, n_channels, endpoint=False)

    fig_polar = go.Figure()
    fig_polar.add_trace(go.Scatterpolar(
        r=cross, theta=angles, fill="toself",
        line=dict(color="steelblue", width=1.5),
        name="Signal",
    ))
    if cross_anom.any():
        anom_angles = angles[cross_anom]
        anom_r = cross[cross_anom]
        fig_polar.add_trace(go.Scatterpolar(
            r=anom_r, theta=anom_angles,
            mode="markers",
            marker=dict(color=highlight_color, size=6, symbol="x"),
            name="Аномалия",
        ))
    fig_polar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=f"Сечение на шаге {inspect_step} (score max={cross_scores.max():.1f})",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_polar, use_container_width=True)

    st.divider()
    st.header("📉 Профиль аномальности по глубине")

    anom_density = anomaly_mask.sum(axis=1).astype(float)
    fig_prof = go.Figure()
    fig_prof.add_trace(go.Bar(
        x=np.arange(n_steps), y=anom_density,
        marker_color="coral", opacity=0.7,
        name="Аномалий/шаг",
    ))
    fig_prof.update_layout(
        title="Плотность аномалий по глубине",
        xaxis_title="Step",
        yaxis_title="Count",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_prof, use_container_width=True)
