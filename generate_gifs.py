# =============================================================
#   generate_gifs.py — Record SUMO simulations as GIFs
#   Uses SUMO's built-in screenshot export (no screen capture)
# =============================================================

import os
import sys
import time
import numpy as np
import imageio
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sumo_rl
import traci
from agents.dqn_agent import DQNAgent
from agents.fixed_timer import FixedTimerAgent
from config import (
    NET_FILE, ROUTE_FILE,
    DELTA_TIME, YELLOW_TIME, MIN_GREEN
)

GIF_DIR     = os.path.join("results", "gifs")
FRAMES_DIR  = os.path.join("results", "frames")
NUM_SECONDS = 300
CAPTURE_EVERY = 2

os.makedirs(GIF_DIR,    exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


def record_agent(agent, agent_name, num_seconds=NUM_SECONDS):
    """Run agent with GUI and capture frames using SUMO screenshot."""
    print(f"\n Recording {agent_name}...")

    frames_subdir = os.path.join(FRAMES_DIR, agent_name.replace(" ", "_"))
    os.makedirs(frames_subdir, exist_ok=True)

    env = sumo_rl.SumoEnvironment(
        net_file      = NET_FILE,
        route_file    = ROUTE_FILE,
        num_seconds   = num_seconds,
        delta_time    = DELTA_TIME,
        yellow_time   = YELLOW_TIME,
        min_green     = MIN_GREEN,
        use_gui       = True,
        sumo_warnings = False,
        single_agent  = True,
    )

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs  = reset_result
        info = {}

    state  = np.array(obs, dtype=np.float32).flatten()
    done   = False
    step   = 0
    frames = []

    print(f" Simulation running and capturing screenshots...")
    print(f" DO NOT close the SUMO window!")

    # Auto-zoom and center the SUMO GUI on the intersection
    try:
        traci.gui.setZoom("View #0", 600)
        traci.gui.setOffset("View #0", 300, 300)
        time.sleep(2)
    except Exception as e:
        print(f"  Zoom warning: {e}")

    while not done:
        action      = agent.select_action(state)
        step_result = env.step(action)

        if len(step_result) == 5:
            next_obs, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, _, done, info = step_result

        # Capture screenshot every N steps using SUMO's API
        if step % CAPTURE_EVERY == 0:
            screenshot_path = os.path.join(
                frames_subdir, f"frame_{step:04d}.png"
            )
            try:
                traci.gui.screenshot("View #0", screenshot_path)
                frames.append(screenshot_path)
                if step % 50 == 0:
                    print(f"  Captured {len(frames)} frames (step {step})")
            except Exception as e:
                pass

        state = np.array(next_obs, dtype=np.float32).flatten()
        step += 1
        time.sleep(0.02)

    env.close()

    # Stitch frames into GIF
    gif_path = os.path.join(
        GIF_DIR, f"{agent_name.lower().replace(' ', '_')}.gif"
    )
    print(f"\n Stitching {len(frames)} frames into GIF...")

    gif_frames = []
    for fp in frames:
        if os.path.exists(fp):
            img = Image.open(fp).resize((640, 360))
            gif_frames.append(np.array(img))

    if gif_frames:
        imageio.mimsave(gif_path, gif_frames, fps=8, loop=0)
        print(f" Saved → {gif_path}")
    else:
        print(f" No frames captured!")

    return gif_path


if __name__ == "__main__":
    print("=" * 50)
    print("  SUMO GIF RECORDER")
    print("=" * 50)

    # Record Fixed Timer
    input("\nPress ENTER to record FIXED TIMER agent...")
    fixed_agent = FixedTimerAgent()
    fixed_gif   = record_agent(fixed_agent, "Fixed Timer")

    # Record DQN Agent
    input("\nPress ENTER to record DQN agent...")
    dqn_agent  = DQNAgent()
    model_path = os.path.join("saved_models", "dqn_final.pth")
    if os.path.exists(model_path):
        dqn_agent.load(model_path)
        dqn_agent.epsilon = 0.0
    dqn_gif = record_agent(dqn_agent, "DQN Agent")

    print("\n" + "=" * 50)
    print("  BOTH GIFs SAVED!")
    print("=" * 50)
    print(f"  Fixed Timer → {fixed_gif}")
    print(f"  DQN Agent   → {dqn_gif}")