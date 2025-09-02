
from typing import List
from lib_piglet.utils.tools import eprint
import glob, os, sys, time, json
import heapq

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
test_single_level = True
level = 2
test = 7

#########################
# Reimplementing the content in get_path() function and replan() function.
#
# They both return a list of paths. A path is a list of (x,y) location tuples.
# The path should be conflict free.
# Hint, you could use some global variables to reuse many resources across get_path/replan frunction calls.
#########################


# Reservation tables
reserve_vertices_table = {}   # t -> {(x,y): agent_id}
reserve_edges_table = {}  # t -> {((current),(next)): agent_id}
planned_paths = [] # Current paths for all agents (path[t]=(x,y))
agent_order = [] # Planning order
max_timestep_global = 0

# Congestion heuristic parameters
CONGESTION_RADIUS = 2          # spatial radius (Manhattan)
CONGESTION_TIME_WINDOW = 2     # time window around t
CONGESTION_ALPHA = 0.4         # weight for congestion penalty in heuristic

# ========== Tools functions ==========
def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def congestion_heuristic(location, timestep):
    """
    Compute a congestion-aware penalty around location at given timestep using
    current reservation tables. This encourages paths to avoid crowded areas.

    The penalty accumulates reservations in a spatial radius and temporal window
    around (location, timestep), discounted by spatial and temporal distance.
    """
    if not reserve_vertices_table:
        return 0.0
    lx, ly = location
    total_penalty = 0.0
    for dt in range(-CONGESTION_TIME_WINDOW, CONGESTION_TIME_WINDOW + 1):
        t = timestep + dt
        rv = reserve_vertices_table.get(t, {})
        if not rv:
            continue
        time_weight = 1.0 / (1 + abs(dt))
        for (cx, cy) in rv.keys():
            d = abs(cx - lx) + abs(cy - ly)
            if d <= CONGESTION_RADIUS:
                space_weight = 1.0 / (1 + d)
                total_penalty += time_weight * space_weight
    return CONGESTION_ALPHA * total_penalty

def successors(rail, loc, direction):
    """
    Generate actions successors (next action).
    Direction: 0: North, 1: East, 2: South, 3: West.
    """
    x, y = loc
    neighbors = []
    valid = rail.get_transitions(x, y, direction) 
    order = [direction, (direction+1)%4, (direction+3)%4, (direction+2)%4]
    for d in order:
        if valid[d]:
            next_x, next_y = x, y
            if d == 0: next_x -= 1
            elif d == 1: next_y += 1
            elif d == 2: next_x += 1
            elif d == 3: next_y -= 1
            neighbors.append(((next_x, next_y), d))
    # WAIT
    neighbors.append(((x, y), direction))
    return neighbors

def is_reserved(agent_id, current, next, t_next):
    """
    Check vertex conflict, edge conflicts (swap and overlap) at t_next.
    Parameters:
        agent_id: agent id.
        current: current position at time t.
        next: next position at time t_next.
        t_next: next timestep.
    """
    # Vertex conflict
    rv = reserve_vertices_table.get(t_next, {})
    if next in rv and rv[next] != agent_id:
        return True
 
    # Edge conflict
    if t_next-1 in reserve_edges_table:
        re = reserve_edges_table.get(t_next-1, {})
        reverse = (next, current)
        # next to current (swap)
        if reverse in re and re[reverse] != agent_id:
            return True
        forward = (current, next)
        # current to next (overlap)
        if forward in re and re[forward] != agent_id:
            return True
    return False

def reserve_path(agent_id, path, t_start):
    """
    Update reservation table with path from t_start.
    Parameters:
        agent_id: agent id.
        path: generated path per timestep.
        t_start: start timestep for reservation.
    """
    if not path:
        return
    for t in range(t_start, min(len(path)-1, max_timestep_global)):
        current = path[t]
        next = path[t+1]
        reserve_vertices_table.setdefault(t, {})
        reserve_vertices_table.setdefault(t+1, {})
        reserve_edges_table.setdefault(t, {})
        reserve_vertices_table[t].setdefault(current, agent_id)
        reserve_vertices_table[t+1].setdefault(next, agent_id)
        reserve_edges_table[t].setdefault((current, next), agent_id)
    # goal = path[-1]
    # for t in range(len(path), max_timestep_global+1):
    #     reserve_vertices_table.setdefault(t, {})
    #     reserve_vertices_table[t].setdefault(goal, agent_id)

def form_new_path(old_path, new_path, split_t):
    """
    Use split_t as the index, keep the 0~split_t part of old_path, and concatenate the split_t~end part of new_path.

    Parameters:
        old_path: old_path.
        new_path: new_path.
        split_t: failed timestep.
    """
    if split_t <= 0 or not old_path:
        return new_path
    return old_path[:split_t] + new_path[split_t:]

def get_current_position(agent: EnvAgent):
    """
    Get the agent current pose. 
    Return ((x,y), direction).
    """
    loc = getattr(agent, "position", None)
    if loc is None:
        loc = agent.initial_position
    d = getattr(agent, "direction", None)
    if d is None:
        d = agent.initial_direction
    return loc, d

def malfunction_remaining(agent: EnvAgent) -> int:
    """Get remaining malfunction timesteps."""
    malfunction_data = getattr(agent, "malfunction_data", None)
    if malfunction_data and "malfunction" in malfunction_data:
        return int(malfunction_data["malfunction"])

# ---------- Reconstruct path ----------
def reconstruct_path_aligned(parents, goal_state_key, start_location, start_timestep, max_timestep):
    """Reconstruct path from parents."""
    location_by_timestep = {}
    if goal_state_key is not None:
        current_state = goal_state_key
        reversed_states = []
        while current_state is not None:
            reversed_states.append(current_state)
            current_state = parents[current_state]
        reversed_states.reverse()
        
        # Map each timestep to its corresponding position
        for (x, y, d, t) in reversed_states:
            location_by_timestep[t] = (x, y)
        last_timestep = max(location_by_timestep.keys())
        last_location = location_by_timestep[last_timestep]
        # Use last location to fill all timesteps
        for t in range(last_timestep + 1, max_timestep + 1):
            location_by_timestep[t] = last_location
    else:
        # No solution, use WAIT fill all timesteps
        for t in range(start_timestep, max_timestep + 1):
            location_by_timestep[t] = start_location
            
    # If waiting from start, use start location to fill all timesteps
    for t in range(0, start_timestep):
        location_by_timestep.setdefault(t, start_location)
    return [location_by_timestep[t] for t in range(0, max_timestep + 1)]


# ---------- Single-agent Time-based A*  ----------
def single_agent_astar(agent_id, rail, start_location, start_direction, goal, start_timestep, max_t):
    """
    Time-expanded A* with reservation table (single agent vs moving obstacles)

    Parameters:
        agent_id: agent id.
        rail: grid environment.
        start_location: start position.
        start_direction: start direction.
        goal: goal position.
        start_timestep: start timestep (with delayed start).
        max_t: maximum timestep.
    """
    if debug:
        print(f"地图宽高: {rail.height}x{rail.width}")
    open_heap = []
    start_key = (start_location[0], start_location[1], start_direction, start_timestep)
    g_value = {start_key: 0}
    # f = g + h
    h0 = manhattan_distance(start_location, goal) + congestion_heuristic(start_location, start_timestep)
    heapq.heappush(open_heap, (g_value[start_key] + h0, 0, *start_key, None))
    parents = {start_key: None}
    best_goal_state_key = None

    # A*
    while open_heap:
        _, g_val, x, y, direction, timestep, parent = heapq.heappop(open_heap)
        current_key = (x,y,direction,timestep)
        if g_val != g_value.get(current_key, float('inf')):
            continue
        if parents[current_key] is None and parent is not None:
            parents[current_key] = parent
        if (x, y) == goal:
            best_goal_state_key = current_key
            break
        if timestep >= max_t:
            continue
        for (next_location, next_direction) in successors(rail, (x,y), direction):
            next_x, next_y = next_location
            n_timestep = timestep + 1
            if is_reserved(agent_id, (x,y), (next_x,next_y), n_timestep):
                continue
            next_key = (next_x,next_y,next_direction,n_timestep)
            wait_penalty = 0.1 if (next_x == x and next_y == y) else 0
            next_g_val = g_val + 1 + wait_penalty
            if next_g_val < g_value.get(next_key, float('inf')):
                g_value[next_key] = next_g_val
                h_val  = manhattan_distance((next_x,next_y), goal) + congestion_heuristic((next_x, next_y), n_timestep)
                heapq.heappush(open_heap, (next_g_val+h_val, next_g_val, next_x, next_y, next_direction, n_timestep, current_key))
                if next_key not in parents:
                    parents[next_key] = current_key

    # Reconstruct path
    return reconstruct_path_aligned(parents, best_goal_state_key, start_location, start_timestep, max_t)


def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    """
    Multi-agent planning: plan in order and update reservation table.

    Parameters:
        agents: agents list.
        rail: grid environment.
        max_timestep: maximum timestep.
    """
    global reserve_vertices_table, reserve_edges_table, planned_paths, agent_order, max_timestep_global
    # Initialize global variables
    max_timestep_global = int(max_timestep)
    reserve_vertices_table.clear()
    reserve_edges_table.clear()
    
    # Initialize paths for all agents
    planned_paths = [[] for _ in range(len(agents))]

    # Priority rule: deadline (ascending), then by Manhattan distance to goal (ascending)
    priorities = [] # (agent_id, deadline, Manhattan distance)
    for agent_id,agent in enumerate(agents):
        start = agent.initial_position
        goal = agent.target
        ddl = getattr(agent, "deadline", None)
        dist = manhattan_distance(start, goal)
        # ddl - dist
        slack = (ddl - dist) if ddl is not None else sys.maxsize
        priorities.append((agent_id, slack, dist))
        # sort by slack ascending, then by dist ascending
    agent_order = [aid for (aid,_,_) in sorted(priorities, key=lambda x: (x[1], x[2]))]
    #     priorities.append((agent_id, ddl if ddl is not None else max_timestep_global, manhattan_distance(start, goal)))
    # agent_order = [agent_id for (agent_id,_,_) in sorted(priorities, key=lambda x: (x[1], x[2]))]

    for agent_id in agent_order:
        agent = agents[agent_id]
        path_i = single_agent_astar(agent_id, rail, agent.initial_position, agent.initial_direction, agent.target, start_timestep=0, max_t=max_timestep_global)
        planned_paths[agent_id] = path_i
        reserve_path(agent_id, path_i, t_start=0) # update reservation table

    # if debug:
    #     print("=====LOG===== reserve_vertices_table:", file=sys.stderr)
    #     for t in sorted(reserve_vertices_table.keys()):
    #         print(f"  t={t}: {reserve_vertices_table[t]}", file=sys.stderr)
    #     print("=====LOG===== reserve_edges_table:", file=sys.stderr)
    #     for t in sorted(reserve_edges_table.keys()):
    #         print(f"  t={t}: {reserve_edges_table[t]}", file=sys.stderr)

    return planned_paths


def replan(agents: List[EnvAgent], rail: GridTransitionMap, current_timestep: int, 
           existing_paths, max_timestep: int, new_malfunction_agents, failed_agents):
    """
    Cache paths and replan new paths for malfunction agents or failed agents.

    Parameters:
        agents: agents list.
        rail: grid environment.
        current_timestep: at which timestep during the execution, the replan function is called.
        existing_paths: the paths from the previous get_path/replan function call
        max_timestep: maximum timestep.
        new_malfunction_agents: a list of ids of agents that have a new malfunction happens at the current time step.
        failed_agents: a list of ids of agents that failed to reach their intended location at the current timestep.
    """
    global reserve_vertices_table, reserve_edges_table, planned_paths, agent_order, max_timestep_global
    # Initialize global variables
    max_timestep_global = int(max_timestep)
    planned_paths = existing_paths

    # Rebuild reservation table, keep history for t<current_timestep
    reserve_vertices_table = {}
    reserve_edges_table = {}
    current_timestep = int(current_timestep)
    for i in range(len(agents)):
        plan = planned_paths[i]
        for t in range(0, current_timestep):
            current = plan[t]
            next = plan[t+1]
            reserve_vertices_table.setdefault(t, {})
            reserve_vertices_table.setdefault(t+1, {})
            reserve_edges_table.setdefault(t, {})
            reserve_vertices_table[t][current] = i
            reserve_vertices_table[t+1][next] = i
            reserve_edges_table[t][(current, next)] = i

    to_replan = set(new_malfunction_agents) | set(failed_agents) # agents that need replanning

    # Update reservation table for normal agents
    for i in range(len(agents)):
        if i in to_replan:
            continue
        reserve_path(i, planned_paths[i], t_start=current_timestep)

    # Setup priority order
    if not agent_order:
        priorities = []
        for i,agent in enumerate(agents):
            ddl = getattr(agent, "deadline", None)
            dist = manhattan_distance(agent.initial_position, agent.target)
            slack = (ddl - dist) if ddl is not None else sys.maxsize
            priorities.append((i, slack, dist))
        # sort by slack ascending, then by dist ascending
        agent_order = [aid for (aid,_,_) in sorted(priorities, key=lambda x: (x[1], x[2]))]
        #     priorities.append((i, ddl if ddl is not None else max_timestep_global, slack, dist))
        # agent_order = [i for (i,_) in sorted(priorities, key=lambda x: (x[1],x[2], x[3]))]

    # Replan by priority
    for i in [k for k in agent_order if k in to_replan]:
        agent = agents[i]
        cur_loc, cur_dir = get_current_position(agent)
        wait_steps = malfunction_remaining(agent) if i in new_malfunction_agents else 0

        # waiting for malfunction
        for k in range(wait_steps):
            t = current_timestep + k
            reserve_vertices_table.setdefault(t, {})
            reserve_vertices_table[t][cur_loc] = i
            if t-1 >= 0: # edge circle, stay at the same location
                reserve_edges_table.setdefault(t-1, {})
                reserve_edges_table[t-1][(cur_loc, cur_loc)] = i

        new_start_timestep = current_timestep + wait_steps
        new_path = single_agent_astar(i, rail, cur_loc, cur_dir, agent.target, start_timestep=new_start_timestep, max_t=max_timestep_global)
        planned_paths[i] = form_new_path(planned_paths[i], new_path, split_t=current_timestep)
        reserve_path(i, planned_paths[i], t_start=current_timestep)

    # if debug:
    #     print("=====LOG===== replan reserve_vertices_table:", file=sys.stderr)
    #     for t in sorted(reserve_vertices_table.keys()):
    #         print(f"  t={t}: {reserve_vertices_table[t]}", file=sys.stderr)
    #     print("=====LOG===== replan reserve_edges_table:", file=sys.stderr)
    #     for t in sorted(reserve_edges_table.keys()):
    #         print(f"  t={t}: {reserve_edges_table[t]}", file=sys.stderr)

    return planned_paths

#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv, replan = replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))

        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"multi_test_case/level{}_test_{}.pkl".format(level, test)))
        elif test_single_level:
            test_cases = glob.glob(os.path.join(script_path, f"multi_test_case/level{level}_test_*.pkl"))
        test_cases.sort()
        deadline_files =  [test.replace(".pkl",".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan = replan)




