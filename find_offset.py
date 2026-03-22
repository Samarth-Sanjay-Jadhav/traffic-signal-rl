import traci
import sumo_rl
import sys
sys.path.insert(0, '.')
from config import NET_FILE, ROUTE_FILE, DELTA_TIME, YELLOW_TIME, MIN_GREEN

env = sumo_rl.SumoEnvironment(
    net_file     = NET_FILE,
    route_file   = ROUTE_FILE,
    num_seconds  = 100,
    delta_time   = DELTA_TIME,
    yellow_time  = YELLOW_TIME,
    min_green    = MIN_GREEN,
    use_gui      = True,
    single_agent = True,
)
env.reset()

# Get junction position
pos = traci.junction.getPosition("C")
print(f"\n Junction 'C' position: x={pos[0]:.2f}, y={pos[1]:.2f}")
print(f" Use these values in setOffset!")

env.close()