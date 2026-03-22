# =============================================================
#   agents/fixed_timer.py — Fixed Timer Baseline Agent
#   Switches phase every FIXED_GREEN_TIME seconds regardless
#   of traffic conditions. Used as comparison baseline.
# =============================================================

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIXED_GREEN_TIME, DELTA_TIME


class FixedTimerAgent:
    """
    Fixed-time traffic signal controller (baseline).
    Switches phase every fixed interval — no awareness of traffic.
    This is what most real-world traffic lights do today.
    """

    def __init__(self):
        self.timer        = 0       # Time elapsed on current phase
        self.current_phase = 0      # Track current phase

        print(f"[FixedTimer] Initialized.")
        print(f"             Green time : {FIXED_GREEN_TIME}s per phase")

    def select_action(self, state=None):
        """
        Ignore state entirely.
        Switch phase when timer exceeds FIXED_GREEN_TIME.
        """
        self.timer += DELTA_TIME

        if self.timer >= FIXED_GREEN_TIME:
            self.timer = 0
            return 1   # Switch phase

        return 0       # Keep current phase

    def reset(self):
        """Reset timer at start of each episode."""
        self.timer         = 0
        self.current_phase = 0