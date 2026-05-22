import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

st.set_page_config(layout="wide", page_title="NPZ Magnetogram Viewer")

st.title("🧲 NPZ Magnetogram Viewer")

base_dir = r"ROW_DATA\df"

stands = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
sensors = ["mdr", "umdp"]

st.sidebar.header("⚙️ Параметры")

selected_stand = st.sidebar.selectbox("Стенд", stands, key="stand")
selected_sensor = st.sidebar.selectbox("Датчик", sensors, key="sensor")

npz_path = os.path.join(base_dir, selected_stand, selected_sensor, "npz_datasets")
files = sorted(glob.glob(os.path.join(npz_path, "*.npz")))

if not files:
    st.error(f"Файлы не найдены: {npz_path}")
    st.stop()

st.sidebar.divider()
st.sidebar.subheader("📁 Секторы")

show_all = st.sidebar.checkbox("Все секторы", value=False, key="show_all")

sector_labels = [os.path.basename(f) for f in files]
selected_file = st.sidebar.selectbox("Выбрать сектор", sector_labels, key="file")

st.sidebar.divider()
st.sidebar.metric("Всего секторов", len(files))

if show_all:
    all_mag, all_vel, all_acc, all_ori, all_odom = [], [], [], [], []
    offset = 0
    sector_boundaries = []
    for f in files:
        data = np.load(f)
        n = data["magnetogram"].shape[0]
        sector_boundaries.append((offset, offset + n, os.path.basename(f)))
        offset += n
        all_mag.append(data["magnetogram"])
        all_vel.append(data["velocity"])
        all_acc.append(data["accelerometer"])
        all_ori.append(data["orientation"])
        all_odom.append(float(data["odomstep"]))
    magnetogram = np.concatenate(all_mag, axis=0)
    velocity = np.concatenate(all_vel, axis=0)
    accelerometer = np.concatenate(all_acc, axis=0)
    orientation = np.concatenate(all_ori, axis=0)
    odomstep = np.mean(all_odom)
else:
    target = [f for f in files if os.path.basename(f) == selected_file][0]
    data = np.load(target)
    magnetogram = data["magnetogram"]
    velocity = data["velocity"]
    accelerometer = data["accelerometer"]
    orientation = data["orientation"]
    odomstep = float(data["odomstep"])
    sector_boundaries = []

n_steps, n_channels = magnetogram.shape

st.sidebar.divider()
st.sidebar.markdown(f"**Шагов:** {n_steps}")
st.sidebar.markdown(f"**Каналов:** {n_channels}")
st.sidebar.markdown(f"**Odom step:** {odomstep:.6f}")

st.sidebar.divider()
st.sidebar.subheader("🔍 Окно просмотра")

selected_cmap = st.sidebar.selectbox("Colormap", ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "seismic", "RdBu_r", "jet"], index=0, key="cmap")

ch_start = st.sidebar.number_input("Канал от", min_value=0, max_value=n_channels - 1, value=0, key="cs")
ch_end = st.sidebar.number_input("Канал до", min_value=1, max_value=n_channels, value=n_channels, key="ce")
ch_end = max(ch_end, ch_start + 1)

st.sidebar.divider()
st.sidebar.subheader("📊 Обработка")

detrend = st.sidebar.checkbox("Detrend", value=False, key="detrend")
detrend_mode = st.sidebar.radio("Тип", ["По каналам", "По шагам", "Глобальный mean"], index=0, key="dt_mode", horizontal=True)

display_mag = magnetogram.copy()
if detrend:
    if detrend_mode == "По каналам":
        display_mag = display_mag - display_mag.mean(axis=0, keepdims=True)
    elif detrend_mode == "По шагам":
        display_mag = display_mag - display_mag.mean(axis=1, keepdims=True)
    else:
        display_mag = display_mag - display_mag.mean()

st.divider()

st.subheader("📍 Навигация по глубине")

window_size = st.slider("Размер окна (шагов)", 50, min(2000, n_steps), min(500, n_steps), key="win_size", step=50)

position = st.slider("", 0, n_steps - 1, 0, key="position_slider")

left = max(0, position - window_size // 2)
right = min(n_steps, position + window_size // 2)
if right - left < window_size:
    if position < n_steps // 2:
        right = min(n_steps, left + window_size)
    else:
        left = max(0, right - window_size)

actual_width = right - left

progress_pct = (position / (n_steps - 1)) * 100 if n_steps > 1 else 0
st.caption(f"Позиция: шаг {position} ({progress_pct:.1f}%) | Окно: {left}–{right-1} | Размер: {actual_width} шагов")

mag_window = display_mag[left:right, ch_start:ch_end]
vel_window = velocity[left:right]
acc_window = accelerometer[left:right, :]

fig, ax = plt.subplots(figsize=(22, 6))
im = ax.imshow(mag_window, aspect="auto", cmap=selected_cmap, origin="lower",
               extent=[ch_start, ch_end, left, right])

if sector_boundaries:
    for (b_start, b_end_val, fname) in sector_boundaries:
        if left < b_end_val < right:
            ax.axhline(y=b_end_val, color="white", linewidth=2, linestyle="--", alpha=0.8)
            ax.text(ch_end + (ch_end - ch_start) * 0.01, b_end_val, f"  {fname[:14]}",
                    color="white", fontsize=8, va="center", alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

cursor_offset = position - left
ax.axhline(y=position, color="red", linewidth=2, alpha=0.7)
ax.text(ch_start - (ch_end - ch_start) * 0.02, position, f" ← шаг {position}",
        color="red", fontsize=9, va="center", fontweight="bold")

ax.set_xlabel("Channel")
ax.set_ylabel("Step (глубина)")
ax.set_title(f"Magnetogram: окно {left}–{right-1}")
cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Value")
st.pyplot(fig)
plt.close()

st.divider()

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Магнитометр — Каналы")
    plot_ch_count = st.slider("Кол-во каналов", 1, min(n_channels, 1024), min(32, n_channels), key="ch_cnt")
    fig2, ax2 = plt.subplots(figsize=(16, 3))
    for i in range(plot_ch_count):
        ch_idx = ch_start + (i % (ch_end - ch_start))
        if ch_idx < n_channels:
            ax2.plot(display_mag[:, ch_idx], alpha=0.4, linewidth=0.5)
    ax2.axvline(x=position, color="red", linewidth=2, alpha=0.7, linestyle="--")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Value")
    ax2.set_title(f"Channels {ch_start}–{min(ch_start + plot_ch_count - 1, n_channels - 1)}")
    st.pyplot(fig2)
    plt.close()

with col_right:
    st.subheader("Скорость")
    fig3, ax3 = plt.subplots(figsize=(16, 3))
    ax3.plot(velocity, color="coral", linewidth=0.7)
    ax3.axvline(x=position, color="red", linewidth=2, alpha=0.7, linestyle="--")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Velocity")
    st.pyplot(fig3)
    plt.close()

tab1, tab2, tab3 = st.tabs(["Акселерометр", "Ориентация", "Статистика"])

with tab1:
    fig4, ax4 = plt.subplots(figsize=(20, 3))
    ax4.plot(accelerometer[:, 0], label="X", linewidth=0.7)
    ax4.plot(accelerometer[:, 1], label="Y", linewidth=0.7)
    ax4.plot(accelerometer[:, 2], label="Z", linewidth=0.7)
    ax4.axvline(x=position, color="red", linewidth=2, alpha=0.7, linestyle="--")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Acceleration")
    ax4.legend()
    st.pyplot(fig4)
    plt.close()

with tab2:
    unique, counts = np.unique(orientation, return_counts=True)
    fig5, ax5 = plt.subplots(figsize=(20, 3))
    ax5.bar(range(len(unique)), counts, color="mediumseagreen")
    ax5.set_xlabel("Index")
    ax5.set_ylabel("Count")
    ax5.set_title(f"Orientation Distribution ({len(unique)} unique values)")
    ax5.set_xticks(range(len(unique)))
    ax5.set_xticklabels([str(u) for u in unique], rotation=45)
    st.pyplot(fig5)
    plt.close()

with tab3:
    stat_mag = display_mag if detrend else magnetogram
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Min магнитометра", int(stat_mag.min()))
    col_s2.metric("Max магнитометра", int(stat_mag.max()))
    col_s3.metric("Mean магнитометра", f"{stat_mag.mean():.1f}")
    col_s4, col_s5, col_s6 = st.columns(3)
    col_s4.metric("Min скорости", int(velocity.min()))
    col_s5.metric("Max скорости", int(velocity.max()))
    col_s6.metric("Mean скорости", f"{velocity.mean():.1f}")
