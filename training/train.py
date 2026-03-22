# =============================================================
#   training/train.py — Main Training Loop (Fixed)
# =============================================================

import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sumo_rl
from agents.dqn_agent import DQNAgent
from config import (
    NET_FILE, ROUTE_FILE, RESULTS_DIR,
    NUM_SECONDS, DELTA_TIME, YELLOW_TIME, MIN_GREEN,
    NUM_EPISODES, SAVE_MODEL_EVERY,
    TARGET_UPDATE, LOG_INTERVAL, BATCH_SIZE
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "saved_models"), exist_ok=True)


def is_done(done):
    """Handle done flag whether it's a bool or a dict."""
    if isinstance(done, dict):
        return done.get('__all__', False)
    return bool(done)


def get_state(obs):
    """Convert sumo-rl observation to flat numpy array."""
    if isinstance(obs, dict):
        arrays = []
        for v in obs.values():
            arr = np.array(v, dtype=np.float32).flatten()
            arrays.append(arr)
        return np.concatenate(arrays) if arrays else np.zeros(27, dtype=np.float32)
    return np.array(obs, dtype=np.float32).flatten()


def get_info_value(info, key):
    """Extract scalar value from info dict (handles nested dicts)."""
    if info is None:
        return 0.0
    val = info.get(key, 0)
    if isinstance(val, dict):
        return sum(val.values())
    return float(val)


def compute_reward(info):
    """Reward from IntelliLight paper: penalize queue and wait time."""
    if info is None:
        return 0.0
    queue = get_info_value(info, 'agents_total_stopped')
    wait  = get_info_value(info, 'agents_total_accumulated_waiting_time') / 100.0
    return -0.25 * queue - 0.25 * wait


def run_training():
    print("=" * 55)
    print("  Autonomous Traffic Signal Control — DQN Training")
    print("=" * 55)

    agent = DQNAgent()

    log_path = os.path.join(RESULTS_DIR, "dqn_training_log.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'total_reward', 'avg_queue',
            'avg_wait', 'epsilon', 'steps', 'avg_loss'
        ])

    for episode in range(1, NUM_EPISODES + 1):

        env = sumo_rl.SumoEnvironment(
            net_file      = NET_FILE,
            route_file    = ROUTE_FILE,
            num_seconds   = NUM_SECONDS,
            delta_time    = DELTA_TIME,
            yellow_time   = YELLOW_TIME,
            min_green     = MIN_GREEN,
            use_gui       = False,
            sumo_warnings = False,
            single_agent  = True,   # ← KEY FIX: single agent mode
        )

        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs  = reset_result
            info = {}

        state        = np.array(obs, dtype=np.float32).flatten()
        total_reward = 0.0
        total_queue  = 0.0
        total_wait   = 0.0
        total_loss   = 0.0
        loss_count   = 0
        step_count   = 0
        done         = False

        while not done:
            action      = agent.select_action(state)
            step_result = env.step(action)   # single_agent=True → action is scalar

            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                next_obs, reward, done_raw, info = step_result
                done = is_done(done_raw)
            else:
                break

            next_state = np.array(next_obs, dtype=np.float32).flatten()

            # Use environment reward directly + our penalty
            queue  = get_info_value(info, 'agents_total_stopped')
            wait   = get_info_value(info, 'agents_total_accumulated_waiting_time') / 100.0
            reward = -0.25 * queue - 0.25 * wait

            agent.store_experience(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            total_reward += reward
            total_queue  += queue
            total_wait   += wait
            step_count   += 1
            state         = next_state

        env.close()

        agent.decay_epsilon()

        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        avg_queue = total_queue / max(step_count, 1)
        avg_wait  = total_wait  / max(step_count, 1)
        avg_loss  = total_loss  / max(loss_count, 1)

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, round(total_reward, 3),
                round(avg_queue, 3), round(avg_wait, 3),
                round(agent.epsilon, 4), step_count,
                round(avg_loss, 6)
            ])

        if episode % LOG_INTERVAL == 0:
            print(f"Episode {episode:>4}/{NUM_EPISODES} | "
                  f"Reward: {total_reward:>8.2f} | "
                  f"Queue: {avg_queue:>6.2f} | "
                  f"Wait: {avg_wait:>7.2f} | "
                  f"Steps: {step_count:>4} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.5f}")

        if episode % SAVE_MODEL_EVERY == 0:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "saved_models"
            )
            agent.save(os.path.join(model_dir, f"dqn_ep{episode}.pth"))

    print("\n Training complete!")
    print(f" Results saved -> {log_path}")

    model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "saved_models"
    )
    agent.save(os.path.join(model_dir, "dqn_final.pth"))
    return agent


if __name__ == "__main__":
    run_training()