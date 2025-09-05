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
test_single_level = False
level = 1
test = 0
# 0,6
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

# ========== Tools functions ==========
def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def successors(rail, loc, direction):
    """
    Generate actions successors (next action).
    Direction: 0: North, 1: East, 2: South, 3: West.
    """
    x, y = loc
    neighbors = []
    valid = rail.get_transitions(x, y, direction) 
    for d in range(4):
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

    # Edge conflict (swap)
    re = reserve_edges_table.get(t_next - 1, {})
    reverse = (next, current)
    if reverse in re and re[reverse] != agent_id:
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
    goal = path[-1]
    for t in range(len(path), max_timestep_global+1):
        reserve_vertices_table.setdefault(t, {})
        reserve_vertices_table[t].setdefault(goal, agent_id)

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
    return 0

def in_bounds(rail, pos):
    if pos is None: 
        return False
    x, y = pos
    return 0 <= x < rail.height and 0 <= y < rail.width

def safe_grid_value(grid, pos, default=10**9):
    # grid is dist_map（height x width）
    if pos is None:
        return default
    x, y = pos
    if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
        return grid[x][y]
    return default


from collections import deque

# Global cache for distance maps
distance_map_cache = {}

def get_distance_map(rail, goal):
    global distance_map_cache
    key = (goal[0], goal[1]) if goal is not None else ("none","none")
    if key in distance_map_cache:
        return distance_map_cache[key]

    dist_map = [[10**9] * rail.width for _ in range(rail.height)]
    if goal is None or not in_bounds(rail, goal):
        distance_map_cache[key] = dist_map
        return dist_map

    q = deque()
    dist_map[goal[0]][goal[1]] = 0
    q.append(goal)
    
    while q:
        x, y = q.popleft()
        for direction in range(4):
            transitions = rail.get_transitions(x, y, direction)
            for d in range(4):
                if transitions[d]:
                    nx, ny = x, y
                    if d == 0: nx -= 1
                    elif d == 1: ny += 1
                    elif d == 2: nx += 1
                    elif d == 3: ny -= 1
                    if nx < 0 or nx >= rail.height or ny < 0 or ny >= rail.width:
                        continue
                    if dist_map[nx][ny] > dist_map[x][y] + 1:
                        dist_map[nx][ny] = dist_map[x][y] + 1
                        q.append((nx, ny))
    
    distance_map_cache[key] = dist_map
    return dist_map

def calculate_agent_priority(agent_id, agent, rail, current_timestep=0):
    """
    Calculate priority for agent planning order.
    Returns tuple for sorting: (slack_time, deadline, estimated_distance)
    Lower values = higher priority
    """
    if current_timestep == 0:
        start = agent.initial_position
    else:
        start, _ = get_current_position(agent)
    
    goal = agent.target
    deadline = getattr(agent, "deadline", max_timestep_global)
    
    # Get minimum distance to goal
    dist_map = get_distance_map(rail, goal)
    min_path_length = safe_grid_value(dist_map, start, default=manhattan_distance(start, goal))
    
    # Calculate slack time (negative means likely to be late)
    slack = deadline - (current_timestep + min_path_length)
    
    return (slack, deadline, min_path_length)

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


# ---------- Single-agent Time-based A* with deadline awareness ----------
def single_agent_astar(agent_id, rail, start_location, start_direction, goal, deadline, start_timestep, max_t):
    """
    Time-expanded A* with reservation table and deadline awareness

    Parameters:
        agent_id: agent id.
        rail: grid environment.
        start_location: start position.
        start_direction: start direction.
        goal: goal position.
        deadline: agent's deadline.
        start_timestep: start timestep (with delayed start).
        max_t: maximum timestep.
    """
    dist_map = get_distance_map(rail, goal)
    
    def deadline_aware_heuristic(pos, timestep):
        """Heuristic that considers both distance and deadline pressure"""
        spatial_dist = safe_grid_value(dist_map, pos, default=manhattan_distance(pos, goal))
        
        # If we're likely to miss deadline, add penalty equivalent to delay cost
        estimated_arrival = timestep + spatial_dist
        if estimated_arrival > deadline:
            # Penalty is 2 * delayed_timesteps according to assignment
            delay_penalty = 2 * (estimated_arrival - deadline)
            return spatial_dist + delay_penalty
        
        # Small bonus for being early (encourages faster completion)
        early_bonus = max(0, deadline - estimated_arrival) * 0.1
        return spatial_dist - early_bonus
    
    open_heap = []
    start_key = (start_location[0], start_location[1], start_direction, start_timestep)
    g_value = {start_key: 0}
    # f = g + h
    heapq.heappush(open_heap, (g_value[start_key] + deadline_aware_heuristic(start_location, start_timestep), 0, *start_key, None))
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
            next_g_val = g_val + 1
            if next_g_val < g_value.get(next_key, float('inf')):
                g_value[next_key] = next_g_val
                h_val = deadline_aware_heuristic((next_x, next_y), n_timestep)
                heapq.heappush(open_heap, (next_g_val+h_val, next_g_val, next_x, next_y, next_direction, n_timestep, current_key))
                if next_key not in parents:
                    parents[next_key] = current_key

    # Reconstruct path
    return reconstruct_path_aligned(parents, best_goal_state_key, start_location, start_timestep, max_t)


def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    """
    Multi-agent planning with deadline-aware priority ordering.

    Parameters:
        agents: agents list.
        rail: grid environment.
        max_timestep: maximum timestep.
    """
    global distance_map_cache
    distance_map_cache = {}
    global reserve_vertices_table, reserve_edges_table, planned_paths, agent_order, max_timestep_global
    # Initialize global variables
    max_timestep_global = int(max_timestep)
    reserve_vertices_table.clear()
    reserve_edges_table.clear()
    
    # Initialize paths for all agents
    planned_paths = [[] for _ in range(len(agents))]

    # Priority: agents with least slack time first (most urgent)
    priorities = []
    for agent_id, agent in enumerate(agents):
        priority_tuple = calculate_agent_priority(agent_id, agent, rail, current_timestep=0)
        priorities.append((agent_id, *priority_tuple))
    
    # Sort by slack (ascending), then deadline (ascending), then distance (ascending)
    agent_order = [agent_id for (agent_id, slack, deadline, dist) in sorted(priorities, key=lambda x: (x[1], x[2], x[3]))]

    for agent_id in agent_order:
        agent = agents[agent_id]
        deadline = getattr(agent, "deadline", max_timestep_global)
        path_i = single_agent_astar(
            agent_id, 
            rail, 
            agent.initial_position, 
            agent.initial_direction, 
            agent.target, 
            deadline,
            start_timestep=0, 
            max_t=max_timestep_global
        )
        planned_paths[agent_id] = path_i
        reserve_path(agent_id, path_i, t_start=0) # update reservation table
        
    return planned_paths


def replan(agents: List[EnvAgent], rail: GridTransitionMap, current_timestep: int, 
           existing_paths, max_timestep: int, new_malfunction_agents, failed_agents):
    """
    Deadline-aware replanning for malfunction agents or failed agents.

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
    current_timestep = int(current_timestep)

    # Rebuild reservation table, keep history for t<current_timestep
    reserve_vertices_table = {}
    reserve_edges_table = {}
    
    for i in range(len(agents)):
        plan = planned_paths[i]
        for t in range(0, current_timestep):
            if t < len(plan) - 1:
                current = plan[t]
                next = plan[t+1]
                reserve_vertices_table.setdefault(t, {})
                reserve_vertices_table.setdefault(t+1, {})
                reserve_edges_table.setdefault(t, {})
                reserve_vertices_table[t][current] = i
                reserve_vertices_table[t+1][next] = i
                reserve_edges_table[t][(current, next)] = i

    to_replan = set(new_malfunction_agents) | set(failed_agents) # agents that need replanning

    # Update reservation table for normal agents (not being replanned)
    for i in range(len(agents)):
        if i in to_replan:
            continue
        reserve_path(i, planned_paths[i], t_start=current_timestep)

    # Calculate priorities for replanning - focus on most urgent agents first
    priorities = []
    for i in range(len(agents)):
        if i in to_replan:
            priority_tuple = calculate_agent_priority(i, agents[i], rail, current_timestep)
            priorities.append((i, *priority_tuple))
    
    # Sort by urgency: negative slack first (already late), then by slack, deadline, distance
    replan_order = [i for (i, slack, deadline, dist) in sorted(priorities, key=lambda x: (x[1], x[2], x[3]))]

    # Replan by priority
    for i in replan_order:
        agent = agents[i]
        cur_loc, cur_dir = get_current_position(agent)
        deadline = getattr(agent, "deadline", max_timestep_global)
        wait_steps = malfunction_remaining(agent) if i in new_malfunction_agents else 0

        # Reserve waiting positions during malfunction
        for k in range(wait_steps):
            t = current_timestep + k
            reserve_vertices_table.setdefault(t, {})
            reserve_vertices_table[t][cur_loc] = i
            if t > 0: # edge reservation, stay at the same location
                reserve_edges_table.setdefault(t-1, {})
                reserve_edges_table[t-1][(cur_loc, cur_loc)] = i

        new_start_timestep = current_timestep + wait_steps
        new_path = single_agent_astar(
            i, 
            rail, 
            cur_loc, 
            cur_dir, 
            agent.target, 
            deadline,
            start_timestep=new_start_timestep, 
            max_t=max_timestep_global
        )
        planned_paths[i] = form_new_path(planned_paths[i], new_path, split_t=current_timestep)
        reserve_path(i, planned_paths[i], t_start=current_timestep)

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