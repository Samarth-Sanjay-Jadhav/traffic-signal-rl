# =============================================================
#   dashboard.py — Streamlit Web Dashboard
#   Run: streamlit run dashboard.py
# =============================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Traffic Signal RL Dashboard",
    page_icon  = "🚦",
    layout     = "wide"
)

# ── Title ─────────────────────────────────────────────────────
st.title("🚦 Autonomous Traffic Signal Control")
st.subheader("Deep Q-Network (DQN) vs Fixed Timer Baseline")
st.markdown("---")

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_training_log():
    path = os.path.join("results", "dqn_training_log.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_eval_results():
    path = os.path.join("results", "evaluation_results.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

train_df = load_training_log()
eval_df  = load_eval_results()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("📋 Project Info")
st.sidebar.markdown("""
**Algorithm:** Deep Q-Network (DQN)

**Simulator:** SUMO

**Framework:** PyTorch + sumo-rl

**Intersection:** 4-way single intersection

**Traffic:** Unbalanced (800 vs 100 cars/hr)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**State Space (27 features):**
- Queue length per lane
- Vehicle count per lane
- Waiting time per lane
- Current phase

**Action Space:**
- 0 = Keep current phase
- 1 = Switch to next phase

**Reward:**
- -0.25 × Queue - 0.25 × Wait
""")

st.sidebar.markdown("---")
st.sidebar.markdown("📚 Based on [IntelliLight (KDD 2018)](https://www.researchgate.net/publication/326504263)")

# ── Key Metrics ───────────────────────────────────────────────
st.markdown("## 📊 Key Results")

if eval_df is not None:
    dqn_data   = eval_df[eval_df['agent'] == 'DQN']
    fixed_data = eval_df[eval_df['agent'] == 'Fixed']

    dqn_reward   = dqn_data['total_reward'].mean()
    fixed_reward = fixed_data['total_reward'].mean()
    dqn_queue    = dqn_data['avg_queue'].mean()
    fixed_queue  = fixed_data['avg_queue'].mean()
    dqn_wait     = dqn_data['avg_wait'].mean()
    fixed_wait   = fixed_data['avg_wait'].mean()

    queue_improvement = ((fixed_queue - dqn_queue) / fixed_queue) * 100
    wait_improvement  = ((fixed_wait  - dqn_wait)  / fixed_wait)  * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label    = "🏆 Avg Reward (DQN)",
            value    = f"{dqn_reward:.2f}",
            delta    = f"{dqn_reward - fixed_reward:.2f} vs Fixed"
        )
    with col2:
        st.metric(
            label    = "🚗 Avg Queue (DQN)",
            value    = f"{dqn_queue:.3f}",
            delta    = f"-{queue_improvement:.1f}% vs Fixed",
            delta_color = "inverse"
        )
    with col3:
        st.metric(
            label    = "⏱️ Avg Wait Time (DQN)",
            value    = f"{dqn_wait:.3f}",
            delta    = f"-{wait_improvement:.1f}% vs Fixed",
            delta_color = "inverse"
        )
    with col4:
        st.metric(
            label    = "📈 Episodes Trained",
            value    = f"{len(train_df) if train_df is not None else 'N/A'}",
            delta    = "Unbalanced traffic"
        )

st.markdown("---")

# ── Training Progress ─────────────────────────────────────────
st.markdown("## 📈 Training Progress")

if train_df is not None:
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x    = train_df['episode'],
            y    = train_df['total_reward'],
            mode = 'lines',
            name = 'Total Reward',
            line = dict(color='#2196F3', width=2)
        ))
        # Add smoothed trend line
        train_df['reward_smooth'] = train_df['total_reward'].rolling(5).mean()
        fig.add_trace(go.Scatter(
            x    = train_df['episode'],
            y    = train_df['reward_smooth'],
            mode = 'lines',
            name = 'Smoothed (5-ep)',
            line = dict(color='#FF5722', width=2, dash='dash')
        ))
        fig.update_layout(
            title  = "Reward per Episode",
            xaxis_title = "Episode",
            yaxis_title = "Total Reward",
            template    = "plotly_dark",
            height      = 350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x    = train_df['episode'],
            y    = train_df['avg_queue'],
            mode = 'lines',
            name = 'Avg Queue',
            line = dict(color='#4CAF50', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x    = train_df['episode'],
            y    = train_df['epsilon'],
            mode = 'lines',
            name = 'Epsilon',
            line = dict(color='#FF9800', width=2),
            yaxis = 'y2'
        ))
        fig2.update_layout(
            title  = "Queue Length + Epsilon Decay",
            xaxis_title = "Episode",
            yaxis_title = "Avg Queue Length",
            yaxis2 = dict(
                title    = "Epsilon",
                overlaying = 'y',
                side     = 'right',
                range    = [0, 1]
            ),
            template = "plotly_dark",
            height   = 350
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── DQN vs Fixed Timer Comparison ────────────────────────────
st.markdown("## ⚔️ DQN Agent vs Fixed Timer Baseline")

if eval_df is not None:
    col1, col2 = st.columns(2)

    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name = 'DQN Agent',
            x    = ['Total Reward', 'Avg Queue', 'Avg Wait'],
            y    = [abs(dqn_reward), dqn_queue, dqn_wait],
            marker_color = '#2196F3'
        ))
        fig3.add_trace(go.Bar(
            name = 'Fixed Timer',
            x    = ['Total Reward', 'Avg Queue', 'Avg Wait'],
            y    = [abs(fixed_reward), fixed_queue, fixed_wait],
            marker_color = '#FF5722'
        ))
        fig3.update_layout(
            title    = "Performance Comparison (lower = better)",
            barmode  = 'group',
            template = "plotly_dark",
            height   = 350
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        episodes  = eval_df['episode'].unique()
        dqn_ep    = eval_df[eval_df['agent'] == 'DQN']['avg_queue'].values
        fixed_ep  = eval_df[eval_df['agent'] == 'Fixed']['avg_queue'].values

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x    = episodes,
            y    = dqn_ep,
            mode = 'lines+markers',
            name = 'DQN Agent',
            line = dict(color='#2196F3', width=2)
        ))
        fig4.add_trace(go.Scatter(
            x    = episodes,
            y    = fixed_ep,
            mode = 'lines+markers',
            name = 'Fixed Timer',
            line = dict(color='#FF5722', width=2)
        ))
        fig4.update_layout(
            title       = "Queue Length per Eval Episode",
            xaxis_title = "Episode",
            yaxis_title = "Avg Queue Length",
            template    = "plotly_dark",
            height      = 350
        )
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Architecture ──────────────────────────────────────────────
st.markdown("## 🧠 DQN Architecture")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("""
    **Input Layer**
    27 features:
    - Queue per lane
    - Vehicles per lane
    - Wait time per lane
    - Current phase
    """)
with col2:
    st.info("""
    **Hidden Layers**
    Dense(64) → ReLU
    Dense(64) → ReLU

    Optimizer: Adam
    Loss: MSE
    """)
with col3:
    st.info("""
    **Output Layer**
    2 Q-values:
    - Action 0: Keep phase
    - Action 1: Switch phase
    """)

st.markdown("---")

# ── Side by Side GIF Comparison ──────────────────────────────
st.markdown("---")
st.markdown("## 🎬 Live Simulation Comparison")
st.markdown("Watch how DQN adapts traffic signals vs Fixed Timer!")

fixed_gif_path = os.path.join("results", "gifs", "fixed_timer.gif")
dqn_gif_path   = os.path.join("results", "gifs", "dqn_agent.gif")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔴 Fixed Timer (Baseline)")
    if os.path.exists(fixed_gif_path):
        with open(fixed_gif_path, "rb") as f:
            st.image(f.read(), caption="Fixed Timer — switches every 30s regardless of traffic", width="stretch")
    else:
        st.warning("Fixed Timer GIF not found. Run generate_gifs.py first!")

with col2:
    st.markdown("### 🟢 DQN Agent (Trained)")
    if os.path.exists(dqn_gif_path):
        with open(dqn_gif_path, "rb") as f:
            st.image(f.read(), caption="DQN Agent — dynamically adapts to traffic conditions", width="stretch")
    else:
        st.warning("DQN GIF not found. Run generate_gifs.py first!")

st.markdown("---")
st.markdown("Built by **Samarth Sanjay Jadhav** | BTech Project 2026 | Based on IntelliLight (KDD 2018)")