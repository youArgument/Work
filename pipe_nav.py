import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
import glob

st.set_page_config(layout="wide", page_title="Pipe Navigator")

st.title("🔧 Pipe Navigator — Навигация по трубе")

base_dir = r"ROW_DATA\df"

stands = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
sensors = ["mdr", "umdp"]

col1, col2 = st.columns(2)
with col1:
    selected_stand = st.selectbox("Стенд", stands)
with col2:
    selected_sensor = st.selectbox("Датчик", sensors)

npz_path = os.path.join(base_dir, selected_stand, selected_sensor, "npz_datasets")
files = sorted(glob.glob(os.path.join(npz_path, "*.npz")))

if not files:
    st.error(f"Файлы не найдены")
    st.stop()

with col1:
    selected_file_idx = st.selectbox("Сектор", range(len(files)), format_func=lambda i: os.path.basename(files[i]))
with col2:
    show_all = st.checkbox("Все секторы", value=False)

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

st.subheader("📍 Позиция по глубине")

view_width = st.slider("Ширина окна (шагов)", 10, min(500, n_steps), min(200, n_steps), key="vw")

current_step = st.slider(
    "Глубина (шаг)", 0, n_steps - 1, 0, key="depth",
    help="Перемещайтесь по трубе"
)

left = max(0, current_step - view_width // 2)
right = min(n_steps, current_step + view_width // 2)
if right - left < view_width:
    if current_step < n_steps // 2:
        right = min(n_steps, left + view_width)
    else:
        left = max(0, right - view_width)

actual_width = right - left
mag_window = magnetogram[left:right, :]
vel_window = velocity[left:right]
acc_window = accelerometer[left:right, :]

st.caption(f"Окно: шаги {left}–{right-1} | Позиция: шаг {current_step} | Всего: {n_steps} шагов, {n_channels} каналов")

st.divider()

st.subheader("📊 Поперечное сечение (текущий шаг)")

cross_mag = magnetogram[current_step, :]
angles_deg = np.linspace(0, 360, n_channels, endpoint=False)

col_left, col_right = st.columns(2)

with col_left:
    fig_cross = go.Figure()
    fig_cross.add_trace(go.Scatterpolar(
        r=cross_mag, theta=angles_deg, fill="toself", fillcolor="steelblue",
        line=dict(color="steelblue"), name="Магнитометр"
    ))
    fig_cross.update_layout(
        polar=dict(radialaxis=dict(visible=True, showgrid=True)),
        title=f"Сечение на шаге {current_step}",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500
    )
    st.plotly_chart(fig_cross, use_container_width=True)

with col_right:
    mean_cross = cross_mag.mean()
    std_cross = cross_mag.std()
    max_ch = np.argmax(cross_mag)
    min_ch = np.argmin(cross_mag)

    st.metric("Среднее", f"{mean_cross:.1f}")
    st.metric("Std", f"{std_cross:.1f}")
    st.metric("Max канал", f"#{max_ch} ({cross_mag[max_ch]})")
    st.metric("Min канал", f"#{min_ch} ({cross_mag[min_ch]})")

    st.markdown("**Акселерометр в этой точке:**")
    st.caption(f"X: {accelerometer[current_step, 0]}  Y: {accelerometer[current_step, 1]}  Z: {accelerometer[current_step, 2]}")
    st.caption(f"Скорость: {velocity[current_step]}")

st.divider()

st.subheader("🗺️ Обзор окна")

cmap_choice = st.selectbox("Colormap", ["Viridis", "Plasma", "Hot", "coolwarm", "Turbo", "Inferno"], index=0)

X = np.arange(actual_width)
Y = np.arange(n_channels)
XX, YY = np.meshgrid(X, Y)

fig_heat = go.Figure(data=go.Heatmap(
    z=mag_window.T, x=X, y=Y, colorscale=cmap_choice,
    colorbar=dict(title="Signal", x=1.02),
    hovertemplate="Step: %{x}<br>Ch: %{y}<br>Val: %{z}<extra></extra>"
))

cursor_y = current_step - left
fig_heat.add_hline(
    y=cursor_y, line=dict(color="red", width=3, dash="dash"),
    annotation_text=f"← Шаг {current_step}", annotation_position="top left"
)

fig_heat.update_layout(
    title=f"Магнитометра: шаги {left}–{right-1}",
    xaxis_title="Step (глубина)",
    yaxis_title="Channel",
    height=600,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

st.subheader("📈 Профиль по глубине")

profile_ch = st.slider("Канал для профиля", 0, n_channels - 1, n_channels // 4, key="profile_ch")

col_p1, col_p2 = st.columns(2)

with col_p1:
    fig_prof = go.Figure()
    fig_prof.add_trace(go.Scatter(
        x=np.arange(actual_width), y=mag_window[:, profile_ch],
        mode="lines", line=dict(color="orange", width=2)
    ))
    fig_prof.add_vline(
        x=cursor_y, line=dict(color="red", width=2, dash="dot"),
        annotation_text=f"Шаг {current_step}"
    )
    fig_prof.update_layout(
        title=f"Канал #{profile_ch} по глубине",
        xaxis_title="Step", yaxis_title="Value", height=350,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_prof, use_container_width=True)

with col_p2:
    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(
        x=np.arange(actual_width), y=vel_window,
        mode="lines", line=dict(color="coral", width=2)
    ))
    fig_vel.add_vline(
        x=cursor_y, line=dict(color="red", width=2, dash="dot")
    )
    fig_vel.update_layout(
        title="Скорость по глубине",
        xaxis_title="Step", yaxis_title="Velocity", height=350,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_vel, use_container_width=True)

st.divider()

st.subheader("🧭 Ориентация трубы (акселерометр)")

fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(x=np.arange(actual_width), y=acc_window[:, 0], mode="lines", name="X", line=dict(width=2)))
fig_acc.add_trace(go.Scatter(x=np.arange(actual_width), y=acc_window[:, 1], mode="lines", name="Y", line=dict(width=2)))
fig_acc.add_trace(go.Scatter(x=np.arange(actual_width), y=acc_window[:, 2], mode="lines", name="Z", line=dict(width=2)))
fig_acc.add_vline(x=cursor_y, line=dict(color="red", width=2, dash="dot"))
fig_acc.update_layout(
    title="Акселерометр 3 оси",
    xaxis_title="Step", yaxis_title="Acceleration", height=350,
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(x=0.95, y=0.95, bgcolor="rgba(0,0,0,0.7)")
)
st.plotly_chart(fig_acc, use_container_width=True)
