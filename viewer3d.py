import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os
import glob

st.set_page_config(layout="wide", page_title="3D Magnetogram Viewer")

st.title("🧲 3D Magnetogram — Ориентация трубы")

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
    st.error(f"Файлы не найдены: {npz_path}")
    st.stop()

with col1:
    selected_file_idx = st.selectbox("Файл (сектор)", range(len(files)), format_func=lambda i: os.path.basename(files[i]))
with col2:
    show_all = st.checkbox("Объединить все секторы", value=False)

if show_all:
    all_mag = []
    all_vel = []
    all_acc = []
    for f in files:
        data = np.load(f)
        all_mag.append(data["magnetogram"])
        all_vel.append(data["velocity"])
        all_acc.append(data["accelerometer"])
    magnetogram = np.concatenate(all_mag, axis=0)
    velocity = np.concatenate(all_vel, axis=0)
    accelerometer = np.concatenate(all_acc, axis=0)
else:
    data = np.load(files[selected_file_idx])
    magnetogram = data["magnetogram"]
    velocity = data["velocity"]
    accelerometer = data["accelerometer"]

n_steps, n_channels = magnetogram.shape
st.markdown(f"**Шагов:** {n_steps}  |  **Каналов:** {n_channels}")

viz_mode = st.radio("Режим 3D", ["Поверхность (развёртка)", "Цилиндр (труба)", "Линии каналов"], horizontal=True)

st.divider()

if viz_mode == "Поверхность (развёртка)":
    st.subheader("Развёртка магнитограммы")
    
    col_a, col_b = st.columns(2)
    with col_a:
        step_start = st.slider("Шаг от", 0, n_steps - 1, 0)
    with col_b:
        step_end = st.slider("Шаг до", 1, n_steps, min(1000, n_steps))
    
    step_end = max(step_end, step_start + 1)
    mag_slice = magnetogram[step_start:step_end, :]
    
    cmap_options = {
        "Viridis": "Viridis", "Plasma": "Plasma", "Inferno": "Inferno",
        "Hot": "Hot", "Coolwarm": "coolwarm", "Jet": "Jet",
        "Cividis": "Cividis", "Turbo": "Turbo"
    }
    selected_cmap = st.selectbox("Colormap", list(cmap_options.keys()), index=0)
    
    X = np.arange(step_end - step_start)
    Y = np.arange(n_channels)
    XX, YY = np.meshgrid(X, Y)
    ZZ = mag_slice.T
    
    fig = go.Figure(data=[go.Surface(
        z=ZZ, x=XX, y=YY, colorscale=cmap_options[selected_cmap],
        colorbar=dict(title="Значение"),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True),
            x=dict(show=True, highlightcolor="white"),
            y=dict(show=True, highlightcolor="white")
        )
    )])
    
    fig.update_layout(
        title=f"Magnetogram Surface: steps {step_start}–{step_end-1}",
        scene=dict(
            xaxis_title="Step (ось трубы)",
            yaxis_title="Channel (угол)",
            zaxis_title="Signal",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        scene_aspectmode="data",
        width=1200, height=700
    )
    st.plotly_chart(fig, use_container_width=True)

elif viz_mode == "Цилиндр (труба)":
    st.subheader("3D Цилиндр — магнитограмма на поверхности трубы")
    
    col_a, col_b = st.columns(2)
    with col_a:
        step_start = st.slider("Шаг от", 0, n_steps - 1, 0)
    with col_b:
        step_end = st.slider("Шаг до", 1, n_steps, min(500, n_steps))
    
    step_end = max(step_end, step_start + 1)
    mag_cyl = magnetogram[step_start:step_end, :]
    
    n_display = mag_cyl.shape[0]
    radius = 1.0
    
    theta = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    z = np.linspace(0, n_display, n_display)
    Theta, Z = np.meshgrid(theta, z)
    
    X_cyl = radius * np.cos(Theta)
    Y_cyl = radius * np.sin(Theta)
    
    mean_val = mag_cyl.mean()
    std_val = mag_cyl.std()
    Z_color = (mag_cyl - mean_val) / (std_val + 1e-8)
    
    cmap_options = ["Viridis", "Plasma", "Inferno", "Hot", "coolwarm", "Jet", "Turbo", "Cividis"]
    selected_cmap = st.selectbox("Colormap", cmap_options, index=0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=X_cyl, y=Y_cyl, z=Z,
        colorscale=selected_cmap,
        surfacecolor=Z_color,
        colorbar=dict(title="Signal (std)", x=1.02),
        opacity=0.95,
        showscale=True
    ))
    
    fig.update_layout(
        title=f"Pipe Cylinder: steps {step_start}–{step_end-1}, {n_channels} channels",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z (ось трубы)",
            aspectmode="data",
            camera=dict(eye=dict(x=2, y=2, z=0.5))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        width=1200, height=700
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Показать осевые сечения"):
        n_slices = st.slider("Кол-во сечений", 1, 8, 4)
        fig2 = go.Figure()
        for i in range(n_slices):
            angle = 2 * np.pi * i / n_slices
            x_slice = radius * np.cos(angle) * np.ones(n_display)
            y_slice = radius * np.sin(angle) * np.ones(n_display)
            channel_idx = int(n_channels * i / n_slices) % n_channels
            fig2.add_trace(go.Scatter3d(
                x=x_slice, y=y_slice, z=z,
                mode="lines",
                name=f"Section {i} (ch {channel_idx})",
                line=dict(width=4)
            ))
        fig2.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            width=900, height=600
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.subheader("Линии отдельных каналов в 3D")
    
    col_a, col_b = st.columns(2)
    with col_a:
        step_start = st.slider("Шаг от", 0, n_steps - 1, 0)
    with col_b:
        step_end = st.slider("Шаг до", 1, n_steps, min(2000, n_steps))
    
    step_end = max(step_end, step_start + 1)
    
    ch_step = st.slider("Шаг каналов (каждый N-й)", 1, max(1, n_channels // 10), max(1, n_channels // 32))
    
    fig = go.Figure()
    for ch in range(0, n_channels, ch_step):
        fig.add_trace(go.Scatter3d(
            x=np.arange(step_end - step_start),
            y=np.full(step_end - step_start, ch),
            z=magnetogram[step_start:step_end, ch],
            mode="lines",
            opacity=0.7,
            showlegend=False,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title=f"Channel Lines: {n_channels // ch_step} channels displayed",
        scene=dict(
            xaxis_title="Step",
            yaxis_title="Channel",
            zaxis_title="Value"
        ),
        width=1200, height=700
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Сопоставление с акселерометром")

tab1, tab2 = st.tabs(["2D графики", "3D траектория"])

with tab1:
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=np.arange(n_steps), y=accelerometer[:, 0], mode="lines", name="Accel X"))
    fig_acc.add_trace(go.Scatter(x=np.arange(n_steps), y=accelerometer[:, 1], mode="lines", name="Accel Y"))
    fig_acc.add_trace(go.Scatter(x=np.arange(n_steps), y=accelerometer[:, 2], mode="lines", name="Accel Z"))
    fig_acc.update_layout(xaxis_title="Step", yaxis_title="Acceleration", width=1200, height=400)
    st.plotly_chart(fig_acc, use_container_width=True)

with tab2:
    x_int = np.cumsum(accelerometer[:, 0])
    y_int = np.cumsum(accelerometer[:, 1])
    z_int = np.arange(n_steps)
    
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter3d(
        x=x_int, y=y_int, z=z_int, mode="lines",
        line=dict(color="cyan", width=2), name="Trajectory"
    ))
    fig_traj.update_layout(
        title="Интегрированная траектория по акселерометру",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z (глубина)"),
        width=1200, height=700
    )
    st.plotly_chart(fig_traj, use_container_width=True)
