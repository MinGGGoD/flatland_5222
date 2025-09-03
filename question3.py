# question3_sipp.py
from typing import List
from lib_piglet.utils.tools import eprint
import glob, os, sys, time, json
import heapq
from functools import lru_cache 

# import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import (
        get_action, Train_Actions, Directions, check_conflict,
        path_controller, evaluator, remote_evaluator
    )
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

#########################
# Debugger and visualizer options
#########################
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
test_single_level = True
level = 6
test = 6

#########################
# Globals
#########################
reserve_vertices_table = {}   # t -> {(x,y): agent_id}
reserve_edges_table = {}      # t -> {((current),(next)): agent_id}
planned_paths = []            # Current paths for all agents (path[t]=(x,y))
agent_order = []              # Planning order
max_timestep_global = 0

# ========== Tools functions ==========
def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def congestion_aware_heuristic(pos, goal, timestep, max_t):
    """
    heuristic function, considering congestion
    """
    base_h = manhattan_distance(pos, goal)
    
    # calculate the congestion penalty around the goal position
    congestion_penalty = 0
    for t in range(timestep, min(timestep + base_h + 5, max_t)):
        rv = reserve_vertices_table.get(t, {})
        # check if the goal position and its surrounding are congested
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_pos = (goal[0] + dx, goal[1] + dy)
                if check_pos in rv:
                    congestion_penalty += 0.1
    
    return base_h + congestion_penalty

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
            nx, ny = x, y
            if d == 0: nx -= 1
            elif d == 1: ny += 1
            elif d == 2: nx += 1
            elif d == 3: ny -= 1
            neighbors.append(((nx, ny), d))
    # WAIT（SIPP 中我们会过滤掉显式 WAIT）
    neighbors.append(((x, y), direction))
    return neighbors

def is_reserved(agent_id, current, nxt, t_next):
    """
    Check vertex conflict, edge conflicts (swap and overlap) at t_next.
    """
    # Vertex conflict
    rv = reserve_vertices_table.get(t_next, {})
    if nxt in rv and rv[nxt] != agent_id:
        return True
    # Edge conflict (t_next-1 的边)
    re = reserve_edges_table.get(t_next - 1, {})
    # swap
    if (nxt, current) in re and re[(nxt, current)] != agent_id:
        return True
    # overlap (同边同向占用)
    if (current, nxt) in re and re[(current, nxt)] != agent_id:
        return True
    return False

def reserve_path(agent_id, path, t_start):
    """
    Update reservation table with path from t_start.
    """
    if not path:
        return
    for t in range(t_start, min(len(path)-1, max_timestep_global)):
        cur = path[t]
        nxt = path[t+1]
        reserve_vertices_table.setdefault(t, {})
        reserve_vertices_table.setdefault(t+1, {})
        reserve_edges_table.setdefault(t, {})
        reserve_vertices_table[t].setdefault(cur, agent_id)
        reserve_vertices_table[t+1].setdefault(nxt, agent_id)
        reserve_edges_table[t].setdefault((cur, nxt), agent_id)

    # clear cache every time reserve table is updated
    build_safe_intervals_for_cell.cache_clear()

def form_new_path(old_path, new_path, split_t):
    """
    Use split_t as the index, keep the 0~split_t part of old_path,
    and concatenate the split_t~end part of new_path.
    """
    if split_t <= 0 or not old_path:
        return new_path
    return old_path[:split_t] + new_path[split_t:]

def get_current_position(agent: EnvAgent):
    """
    Get the agent current pose. Return ((x,y), direction).
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

def reconstruct_path_aligned(parents, goal_state_key, start_location, start_timestep, max_timestep):
    """
    backtrack parents to fill time gaps, ensuring every timestep has a position
    """
    T = int(max_timestep)
    res = [start_location for _ in range(T + 1)]

    if goal_state_key is None:
        return res

    # 回溯状态序列
    seq = []
    cur = goal_state_key
    while cur is not None:
        seq.append(cur)
        cur = parents.get(cur, None)
    seq.reverse()

    last_loc = start_location
    last_t = max(0, min(start_timestep, T))

    for (x, y, d, t) in seq:
        t = int(t)
        if t < 0:
            continue
        if t > T:
            break
        if t > last_t:
            for tt in range(last_t, t):
                res[tt] = last_loc
        res[t] = (x, y)
        last_loc = (x, y)
        last_t = t + 1

    if last_t <= T:
        for tt in range(last_t, T + 1):
            res[tt] = last_loc

    return res

def _merge_into_intervals(blocked_times, max_t):
    """
    build safe intervals from blocked times
    """
    intervals = []
    start = 0
    for bt in blocked_times:
        if bt < 0 or bt > max_t:
            continue
        if bt >= start:
            if bt - 1 >= start:
                intervals.append((start, bt - 1))
            start = bt + 1
    if start <= max_t:
        intervals.append((start, max_t))
    return intervals

@lru_cache(maxsize=None)
def build_safe_intervals_for_cell(cell, max_t):
    """
    build safe intervals from reserve_vertices_table
    """
    blocked = sorted(t for t, occ in reserve_vertices_table.items() if cell in occ)
    return tuple(_merge_into_intervals(blocked, max_t))

def earliest_safe_arrival(agent_id, cur_cell, cur_t, nxt_cell, max_t):
    """
    jump in safe intervals: wait in a safe interval of cur_cell, then move to a safe interval of nxt_cell
    conditions:
      - [cur_t, t_arr-1] must be in a safe interval of cur_cell (waiting)
      - t_arr must be in a safe interval of nxt_cell
      - (cur_cell -> nxt_cell) at t_arr must not conflict with reserved vertices/edges (含 swap/overlap)
    return None if no feasible arrival time
    """
    cur_intervals = build_safe_intervals_for_cell(cur_cell, max_t)
    nxt_intervals = build_safe_intervals_for_cell(nxt_cell, max_t)
    if not cur_intervals or not nxt_intervals:
        return None

    arr0 = cur_t + 1  # earliest arrival time by one step move

    # try to align the arrival time to a safe interval of nxt_cell
    for (L, R) in nxt_intervals:
        t_arr = max(arr0, L)
        if t_arr > R:
            continue
        # if there is a conflict, roll forward
        while t_arr <= R and is_reserved(agent_id, cur_cell, nxt_cell, t_arr):
            t_arr += 1
        if t_arr > R:
            continue

        # check if we can wait in cur_cell safely until t_arr-1
        need_wait = t_arr - 1 - cur_t
        if need_wait <= 0:
            return t_arr
        for (cl, cr) in cur_intervals:
            if cur_t >= cl and (t_arr - 1) <= cr:
                return t_arr

    return None

# ---------- Single-agent SIPP-lite ----------
def single_agent_astar(agent_id, rail, start_location, start_direction, goal, start_timestep, max_t):
    """
    time dimension jumps forward by "earliest reachable"
    """
    if debug:
        print(f"地图宽高: {rail.height}x{rail.width}")

    start_intervals = build_safe_intervals_for_cell(start_location, max_t)
    # if the start location is occupied at start_timestep, align the start time to the first safe time
    if start_intervals:
        t0 = start_timestep
        for (li, ri) in start_intervals:
            if ri < t0:
                continue
            if li > t0:
                start_timestep = li
            break

    open_heap = []
    start_key = (start_location[0], start_location[1], start_direction, start_timestep)
    g_value = {start_key: 0}  # g=elapsed time
    parents = {start_key: None}
    heapq.heappush(open_heap, (g_value[start_key] + congestion_aware_heuristic(start_location, goal, start_timestep, max_t),
                               0, *start_key, None))
    best_goal_state_key = None

    while open_heap:
        _, g_val, x, y, direction, timestep, parent = heapq.heappop(open_heap)
        cur_key = (x, y, direction, timestep)
        if g_val != g_value.get(cur_key, float('inf')):
            continue
        if parents[cur_key] is None and parent is not None:
            parents[cur_key] = parent

        if (x, y) == goal:
            best_goal_state_key = cur_key
            break

        if timestep >= max_t:
            continue

        for (next_location, next_direction) in successors(rail, (x, y), direction):
            nx, ny = next_location
            
            # if it is a waiting action, only consider it when necessary (avoid infinite waiting)
            if (nx, ny) == (x, y):
                # only waiting when there is a conflict and the waiting time is less than 5 steps
                if timestep >= max_t - 5:
                    continue
                wait_t = timestep + 1
                if wait_t <= max_t and not is_reserved(agent_id, (x, y), (x, y), wait_t):
                    nxt_key = (x, y, direction, wait_t)
                    nxt_g = g_val + 1
                    if nxt_g < g_value.get(nxt_key, float('inf')):
                        g_value[nxt_key] = nxt_g
                        h = congestion_aware_heuristic((x, y), goal, wait_t, max_t)
                        heapq.heappush(open_heap, (nxt_g + h, nxt_g, x, y, direction, wait_t, cur_key))
                        if nxt_key not in parents:
                            parents[nxt_key] = cur_key
                continue

            t_arr = earliest_safe_arrival(agent_id, (x, y), timestep, (nx, ny), max_t)
            if t_arr is None or t_arr > max_t:
                continue

            nxt_key = (nx, ny, next_direction, t_arr)
            step_cost = t_arr - timestep
            nxt_g = g_val + step_cost
            if nxt_g < g_value.get(nxt_key, float('inf')):
                g_value[nxt_key] = nxt_g
                h = congestion_aware_heuristic((nx, ny), goal, t_arr, max_t)
                heapq.heappush(open_heap, (nxt_g + h, nxt_g, nx, ny, next_direction, t_arr, cur_key))
                if nxt_key not in parents:
                    parents[nxt_key] = cur_key

    path = reconstruct_path_aligned(parents, best_goal_state_key, start_location, start_timestep, max_t)
    
    # if no path found, try to extend the time limit for the second search
    if best_goal_state_key is None and max_t < 500:
        if debug:
            print(f"Agent {agent_id}: 第一次搜索失败，尝试扩展时间限制...")
        extended_max_t = min(max_t * 2, 500)
        return single_agent_astar(agent_id, rail, start_location, start_direction, goal, start_timestep, extended_max_t)
    
    return path

# ---------- Multi-agent ----------
def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    """
    Multi-agent planning: plan in order and update reservation table.
    """
    global reserve_vertices_table, reserve_edges_table, planned_paths, agent_order, max_timestep_global
    max_timestep_global = int(max_timestep)

    reserve_vertices_table.clear()
    reserve_edges_table.clear()
    # clear cache every time reserve table is updated
    build_safe_intervals_for_cell.cache_clear()

    planned_paths = [[] for _ in range(len(agents))]

    # priority strategy: first by slack ascending, then by distance ascending
    priorities = []
    for aid, agent in enumerate(agents):
        start = agent.initial_position
        goal = agent.target
        ddl = getattr(agent, "deadline", None)
        dist = manhattan_distance(start, goal)
        slack = (ddl - dist) if ddl is not None else sys.maxsize
        priorities.append((aid, slack, dist))
    agent_order = [aid for (aid, _, _) in sorted(priorities, key=lambda x: (x[1], x[2]))]

    for aid in agent_order:
        agent = agents[aid]
        path_i = single_agent_astar(aid, rail, agent.initial_position, agent.initial_direction,
                                    agent.target, start_timestep=0, max_t=max_timestep_global)
        planned_paths[aid] = path_i
        reserve_path(aid, path_i, t_start=0)

    return planned_paths

def replan(agents: List[EnvAgent], rail: GridTransitionMap, current_timestep: int,
           existing_paths, max_timestep: int, new_malfunction_agents, failed_agents):
    """
    Cache paths and replan new paths for malfunction agents or failed agents.
    """
    global reserve_vertices_table, reserve_edges_table, planned_paths, agent_order, max_timestep_global
    max_timestep_global = int(max_timestep)
    planned_paths = existing_paths

    # Rebuild reservation table (history t < current_timestep)
    reserve_vertices_table = {}
    reserve_edges_table = {}
    current_timestep = int(current_timestep)

    for i in range(len(agents)):
        plan = planned_paths[i]
        if not plan or len(plan) < 2:
            continue
        T = min(current_timestep, len(plan) - 1)
        for t in range(0, T):
            cur = plan[t]
            nxt = plan[t+1]
            reserve_vertices_table.setdefault(t, {})[cur] = i
            reserve_vertices_table.setdefault(t+1, {})[nxt] = i
            reserve_edges_table.setdefault(t, {})[(cur, nxt)] = i

    build_safe_intervals_for_cell.cache_clear()

    to_replan = set(new_malfunction_agents) | set(failed_agents)

    # Update reservation table for normal agents
    for i in range(len(agents)):
        if i in to_replan:
            continue
        reserve_path(i, planned_paths[i], t_start=current_timestep)  # 自动 cache_clear

    if not agent_order:
        priorities = []
        for i, agent in enumerate(agents):
            ddl = getattr(agent, "deadline", None)
            dist = manhattan_distance(agent.initial_position, agent.target)
            cur_pos, _ = get_current_position(agent)
            cur_dist = manhattan_distance(cur_pos, agent.target)  
            slack = (ddl - cur_dist) if ddl is not None else sys.maxsize 
            priorities.append((i, slack, dist, cur_dist))
        agent_order = [aid for (aid, _, _, _) in sorted(priorities, key=lambda x: (x[1], x[3]))]

    # Replan by priority
    for i in [k for k in agent_order if k in to_replan]:
        agent = agents[i]
        cur_loc, cur_dir = get_current_position(agent)
        wait_steps = malfunction_remaining(agent) if i in new_malfunction_agents else 0

        # occupy the position during waiting (self-loop edge)
        for k in range(wait_steps):
            t = current_timestep + k
            reserve_vertices_table.setdefault(t, {})[cur_loc] = i
            if t-1 >= 0:
                reserve_edges_table.setdefault(t-1, {})[(cur_loc, cur_loc)] = i

        new_start_timestep = current_timestep + wait_steps

        new_path = single_agent_astar(
            i, rail, cur_loc, cur_dir, agent.target,
            start_timestep=new_start_timestep, max_t=max_timestep_global
        )
        planned_paths[i] = form_new_path(planned_paths[i], new_path, split_t=current_timestep)
        reserve_path(i, planned_paths[i], t_start=current_timestep)  # 自动 cache_clear

    return planned_paths

#####################################################################
# Entrypoint
#####################################################################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv, replan=replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))

        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path, f"multi_test_case/level{level}_test_{test}.pkl"))
        elif test_single_level:
            test_cases = glob.glob(os.path.join(script_path, f"multi_test_case/level{level}_test_*.pkl"))

        test_cases.sort()
        deadline_files = [test.replace(".pkl", ".ddl") for test in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan=replan)
