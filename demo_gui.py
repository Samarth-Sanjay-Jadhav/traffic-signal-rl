# =============================================================
#   demo_gui.py — Visual Demo of Trained DQN Agent
#   Run this to see the DQN agent controlling traffic live!
# =============================================================

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sumo_rl
from agents.dqn_agent import DQNAgent
from agents.fixed_timer import FixedTimerAgent
from config import (
    NET_FILE, ROUTE_FILE,
    NUM_SECONDS, DELTA_TIME, YELLOW_TIME, MIN_GREEN
)


def run_gui_demo(agent, title, num_seconds=300):
    """Run agent with SUMO GUI open."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"  Watch the intersection in the SUMO window!")
    print(f"{'='*50}")
    print("  Press PLAY button in SUMO GUI to start...")

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

    state        = np.array(obs, dtype=np.float32).flatten()
    done         = False
    total_reward = 0.0
    queues       = []
    step         = 0

    while not done:
        action      = agent.select_action(state)
        step_result = env.step(action)

        if len(step_result) == 5:
            next_obs, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, _, done, info = step_result

        queue  = info.get('agents_total_stopped', 0)
        wait   = info.get('agents_total_accumulated_waiting_time', 0) / 100.0
        reward = -0.25 * queue - 0.25 * wait

        total_reward += reward
        queues.append(queue)
        step += 1
        state = np.array(next_obs, dtype=np.float32).flatten()

        if step % 20 == 0:
            print(f"  Step {step:>4} | Queue: {queue:>5.1f} | "
                  f"Reward so far: {total_reward:>8.2f}")

        time.sleep(0.3)

    env.close()

    print(f"\n  DEMO COMPLETE!")
    print(f"  Total Reward : {total_reward:.2f}")
    print(f"  Avg Queue    : {np.mean(queues):.3f}")
    return total_reward, np.mean(queues)


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  TRAFFIC SIGNAL CONTROL — LIVE DEMO")
    print("="*50)
    print("\n  This demo runs for 300 seconds each agent.")
    print("  Watch how DQN adapts vs Fixed Timer!\n")

    input("  Press ENTER to start FIXED TIMER demo...")
    fixed_agent = FixedTimerAgent()
    ft_reward, ft_queue = run_gui_demo(
        fixed_agent,
        "FIXED TIMER AGENT (Baseline)",
        num_seconds=300
    )

    input("\n  Press ENTER to start DQN AGENT demo...")
    dqn_agent = DQNAgent()
    model_path = os.path.join("saved_models", "dqn_final.pth")
    if os.path.exists(model_path):
        dqn_agent.load(model_path)
        dqn_agent.epsilon = 0.0
    else:
        print("  [Warning] No trained model found!")

    dqn_reward, dqn_queue = run_gui_demo(
        dqn_agent,
        "DQN AGENT (Trained)",
        num_seconds=300
    )

    print("\n" + "="*50)
    print("  FINAL COMPARISON")
    print("="*50)
    print(f"  {'Metric':<20} {'Fixed Timer':>12} {'DQN Agent':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Total Reward':<20} {ft_reward:>12.2f} {dqn_reward:>12.2f}")
    print(f"  {'Avg Queue':<20} {ft_queue:>12.3f} {dqn_queue:>12.3f}")

    if dqn_queue < ft_queue:
        improvement = ((ft_queue - dqn_queue) / ft_queue) * 100
        print(f"\n  DQN reduced queue by {improvement:.1f}% vs Fixed Timer!")
    print("="*50)