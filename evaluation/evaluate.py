# =============================================================
#   evaluation/evaluate.py — Compare DQN vs Fixed Timer
# =============================================================

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sumo_rl
from agents.dqn_agent import DQNAgent
from agents.fixed_timer import FixedTimerAgent
from config import (
    NET_FILE, ROUTE_FILE, RESULTS_DIR,
    NUM_SECONDS, DELTA_TIME, YELLOW_TIME,
    MIN_GREEN, EVAL_EPISODES
)

os.makedirs(RESULTS_DIR, exist_ok=True)


def get_state(obs):
    if isinstance(obs, dict):
        return np.concatenate([np.array(v, dtype=np.float32).flatten()
                                for v in obs.values()])
    return np.array(obs, dtype=np.float32).flatten()


def run_episode(agent, use_gui=False):
    """Run one evaluation episode and return metrics."""
    env = sumo_rl.SumoEnvironment(
        net_file      = NET_FILE,
        route_file    = ROUTE_FILE,
        num_seconds   = NUM_SECONDS,
        delta_time    = DELTA_TIME,
        yellow_time   = YELLOW_TIME,
        min_green     = MIN_GREEN,
        use_gui       = use_gui,
        sumo_warnings = False,
	single_agent  = True,
    )

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs  = reset_result
        info = {}

    state        = get_state(obs)
    done         = False
    total_reward = 0.0
    queues       = []
    waits        = []

    while not done:
        action      = agent.select_action(state)
        action_dict = {ts: action for ts in env.ts_ids}
        step_result = env.step(action)

        if len(step_result) == 5:
            next_obs, _, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, _, done, info = step_result

        next_state = get_state(next_obs)

        queues.append(info.get('agents_total_stopped', 0))
        waits.append(info.get('agents_total_accumulated_waiting_time', 0))
        total_reward += -0.25 * queues[-1] - 0.25 * waits[-1] / 100.0
        state = next_state

    env.close()

    return {
        'total_reward' : round(total_reward, 3),
        'avg_queue'    : round(float(np.mean(queues)), 3),
        'avg_wait'     : round(float(np.mean(waits)), 3),
        'max_queue'    : round(float(np.max(queues)), 3),
    }


def evaluate():
    print("=" * 55)
    print("  Evaluation: DQN Agent vs Fixed Timer Baseline")
    print("=" * 55)

    dqn_agent  = DQNAgent()
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models", "dqn_final.pth"
    )

    if os.path.exists(model_path):
        dqn_agent.load(model_path)
        dqn_agent.epsilon = 0.0
    else:
        print(f"[Warning] No saved model found at {model_path}")
        print("[Warning] Evaluating with untrained agent.")

    fixed_agent   = FixedTimerAgent()
    dqn_results   = []
    fixed_results = []

    for ep in range(1, EVAL_EPISODES + 1):
        print(f"\n Episode {ep}/{EVAL_EPISODES}")

        print("  Running DQN agent...")
        dqn_metrics = run_episode(dqn_agent)
        dqn_results.append(dqn_metrics)
        print(f"  DQN   -> Reward: {dqn_metrics['total_reward']:>8.2f} | "
              f"Queue: {dqn_metrics['avg_queue']:>5.2f} | "
              f"Wait: {dqn_metrics['avg_wait']:>7.2f}")

        print("  Running Fixed Timer agent...")
        fixed_agent.reset()
        fixed_metrics = run_episode(fixed_agent)
        fixed_results.append(fixed_metrics)
        print(f"  Fixed -> Reward: {fixed_metrics['total_reward']:>8.2f} | "
              f"Queue: {fixed_metrics['avg_queue']:>5.2f} | "
              f"Wait: {fixed_metrics['avg_wait']:>7.2f}")

    print("\n" + "=" * 55)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<25} {'DQN Agent':>12} {'Fixed Timer':>12}")
    print("-" * 55)

    metrics = ['total_reward', 'avg_queue', 'avg_wait', 'max_queue']
    labels  = ['Avg Total Reward', 'Avg Queue Length',
               'Avg Wait Time',    'Max Queue Length']

    for metric, label in zip(metrics, labels):
        dqn_avg   = np.mean([r[metric] for r in dqn_results])
        fixed_avg = np.mean([r[metric] for r in fixed_results])
        print(f"{label:<25} {dqn_avg:>12.3f} {fixed_avg:>12.3f}")

    csv_path = os.path.join(RESULTS_DIR, "evaluation_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'agent', 'total_reward',
                         'avg_queue', 'avg_wait', 'max_queue'])
        for i, (d, fx) in enumerate(zip(dqn_results, fixed_results), 1):
            writer.writerow([i, 'DQN',   d['total_reward'],
                             d['avg_queue'],  d['avg_wait'],  d['max_queue']])
            writer.writerow([i, 'Fixed', fx['total_reward'],
                             fx['avg_queue'], fx['avg_wait'], fx['max_queue']])

    print(f"\n Results saved -> {csv_path}")
    plot_comparison(dqn_results, fixed_results)


def plot_comparison(dqn_results, fixed_results):
    """Generate comparison charts."""
    metrics = ['total_reward', 'avg_queue', 'avg_wait']
    titles  = ['Total Reward', 'Avg Queue Length', 'Avg Wait Time']
    colors  = [['#2196F3', '#FF5722']] * 3

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('DQN Agent vs Fixed Timer Baseline',
                 fontsize=14, fontweight='bold')

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        dqn_vals   = [r[metric] for r in dqn_results]
        fixed_vals = [r[metric] for r in fixed_results]
        episodes   = range(1, len(dqn_vals) + 1)

        ax.plot(episodes, dqn_vals,   'o-', color=color[0],
                label='DQN Agent',   linewidth=2)
        ax.plot(episodes, fixed_vals, 's-', color=color[1],
                label='Fixed Timer', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "comparison_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f" Plot saved -> {plot_path}")
    plt.close()


if __name__ == "__main__":
    evaluate()