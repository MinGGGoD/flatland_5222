from lib_piglet.utils.tools import eprint
from typing import List, Tuple, Dict
import glob, os, sys, time, json
from heapq import heappush, heappop
from collections import deque

try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import (
        get_action, Train_Actions, Directions,
        check_conflict, path_controller, evaluator, remote_evaluator
    )
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

# Debug and visualizer flags
debug = False
visualizer = False

test_single_instance = False
level = 1
test = 4

# --- Shared A* for single agent with reservations ---
def time_space_astar(agent: EnvAgent,
                     start_time: int,
                     start_pos: Tuple[int,int],
                     start_dir: int,
                     rail: GridTransitionMap,
                     max_timestep: int,
                     reserved_cells: Dict[int, set],
                     reserved_edges: Dict[int, set]) -> List[Tuple[int,int]]:
    goal = agent.target
    def h(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    open_heap = []
    g_map: Dict[Tuple[int,int,int,int], int] = {}
    parent: Dict[Tuple[int,int,int,int], Tuple[Tuple[int,int,int,int], Tuple[int,int]]] = {}
    start_state = (start_pos[0], start_pos[1], start_dir, start_time)
    g_map[start_state] = 0
    heappush(open_heap, (h(start_pos), 0) + start_state)

    while open_heap:
        f, g, x, y, direction, t = heappop(open_heap)
        if (x, y) == goal:
            path = []
            key = (x, y, direction, t)
            while key in parent:
                path.append((key[0], key[1]))
                key, _ = parent[key]
            path.append(start_pos)
            return list(reversed(path))
        if t >= max_timestep:
            continue
        nt = t + 1
        # wait action
        if (reserved_cells.get(nt, set()) & {(x, y)}) == set():
            new_state = (x, y, direction, nt)
            if g + 1 < g_map.get(new_state, 1e9):
                g_map[new_state] = g + 1
                parent[new_state] = ((x, y, direction, t), (x, y))
                heappush(open_heap, (g+1 + h((x,y)), g+1) + new_state)
        # move actions
        transitions = rail.get_transitions(x, y, direction)
        for action, valid in enumerate(transitions):
            if not valid:
                continue
            nx, ny = x, y
            if action == Directions.NORTH:
                nx -= 1
            elif action == Directions.EAST:
                ny += 1
            elif action == Directions.SOUTH:
                nx += 1
            elif action == Directions.WEST:
                ny -= 1
            # reservation checks
            if (nx, ny) in reserved_cells.get(nt, set()):
                continue
            if ((nx, ny), (x, y)) in reserved_edges.get(nt, set()):
                continue
            new_state = (nx, ny, action, nt)
            if g + 1 < g_map.get(new_state, 1e9):
                g_map[new_state] = g + 1
                parent[new_state] = ((x, y, direction, t), (nx, ny))
                heappush(open_heap, (g+1 + h((nx,ny)), g+1) + new_state)
    return []

# --- Main planning functions with Static Priority, Spatial Clustering ---
def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int) -> List[List[Tuple[int,int]]]:
    """
    Plan paths by:
    1. Static priority: sort by (real shortest distance, slack).
    2. Spatial clustering: 4 quadrant batches with offset.
    """
    # 1. compute real shortest distances via BFS
    real_dist = {}
    for aid, agent in enumerate(agents):
        start = agent.initial_position
        goal = agent.target
        visited = {start}
        dq = deque([(start, 0)])
        dist = int(1e9)
        while dq:
            (x,y), d = dq.popleft()
            if (x,y) == goal:
                dist = d
                break
            for action, valid in enumerate(rail.get_transitions(x, y, agent.initial_direction)):
                if not valid: continue
                nx, ny = x, y
                if action == Directions.NORTH: nx -= 1
                elif action == Directions.EAST: ny += 1
                elif action == Directions.SOUTH: nx += 1
                elif action == Directions.WEST: ny -= 1
                if (nx,ny) not in visited:
                    visited.add((nx,ny))
                    dq.append(((nx,ny), d+1))
        real_dist[aid] = dist
    # 2. slack: deadline - real_dist
    slack = {}
    for aid, agent in enumerate(agents):
        dl = getattr(agent, 'deadline', max_timestep)
        slack[aid] = dl - real_dist.get(aid, max_timestep)
    # 3. static priority
    priority_list = sorted(range(len(agents)), key=lambda aid: (real_dist.get(aid, 1e9), slack.get(aid, 1e9)))
    priority_rank = {aid: idx for idx, aid in enumerate(priority_list)}

    # 4. spatial clustering
    try:
        h, w = rail.height, rail.width
    except:
        h = w = 0
    mid_x, mid_y = h/2, w/2
    clusters = {i: [] for i in range(4)}
    for aid, agent in enumerate(agents):
        sx, sy = agent.initial_position
        idx = (1 if sx>=mid_x else 0)*2 + (1 if sy>=mid_y else 0)
        clusters[idx].append(aid)
    delta_t = max(1, (h+w)//4)

    reserved_cells, reserved_edges = {}, {}
    all_paths = [[] for _ in agents]

    def reserve_abs(path, offset):
        for i, cell in enumerate(path):
            t = offset + i
            if t > max_timestep: break
            reserved_cells.setdefault(t, set()).add(cell)
            if i>0:
                reserved_edges.setdefault(t, set()).add((path[i-1], cell))

    # batch planning
    for cid in range(4):
        offset = cid * delta_t
        for aid in sorted(clusters[cid], key=lambda a: priority_rank[a]):
            agent = agents[aid]
            # wait prefix
            prefix = [agent.initial_position]*offset
            for t in range(offset):
                reserved_cells.setdefault(t, set()).add(agent.initial_position)
            sub = time_space_astar(
                agent, offset, agent.initial_position, agent.initial_direction,
                rail, max_timestep, reserved_cells, reserved_edges
            )
            path = prefix + sub
            reserve_abs(sub, offset)
            all_paths[aid] = path
    # pad
    full = max_timestep+1
    for aid, path in enumerate(all_paths):
        if not path:
            all_paths[aid] = [agents[aid].initial_position]*full
        elif len(path)<full:
            all_paths[aid] = path + [path[-1]]*(full-len(path))
    return all_paths

# --- Predictive replanning with Static Priority ---
def replan(agents: List[EnvAgent], rail: GridTransitionMap,
           current_timestep: int, existing_paths: List[List[Tuple[int,int]]],
           max_timestep: int, new_malfunction_agents: List[int], failed_agents: List[int]) -> List[List[Tuple[int,int]]]:
    """
    Predictive replanning: lookahead conflicts and replan conflict set by static priority.
    """
    # Recompute real_dist and slack for priority in replan
    from collections import deque
    real_dist = {}
    for aid, agent in enumerate(agents):
        start = agent.initial_position
        goal = agent.target
        visited = {start}
        dq = deque([(start, 0)])
        dist = int(1e9)
        while dq:
            (x,y), d = dq.popleft()
            if (x,y) == goal:
                dist = d
                break
            for action, valid in enumerate(rail.get_transitions(x, y, agent.initial_direction)):
                if not valid: continue
                nx, ny = x, y
                if action == Directions.NORTH: nx -= 1
                elif action == Directions.EAST: ny += 1
                elif action == Directions.SOUTH: nx += 1
                elif action == Directions.WEST: ny -= 1
                if (nx,ny) not in visited:
                    visited.add((nx,ny))
                    dq.append(((nx,ny), d+1))
        real_dist[aid] = dist
    slack = {}
    for aid, agent in enumerate(agents):
        dl = getattr(agent, 'deadline', max_timestep)
        slack[aid] = dl - real_dist.get(aid, max_timestep)

    priority_list = sorted(range(len(agents)), key=lambda aid: (real_dist.get(aid,1e9), slack.get(aid,1e9)))
    priority_rank = {aid: rank for rank, aid in enumerate(priority_list)}

    PRED = 5
    reserved_cells, reserved_edges = {}, {}
    # build predicted positions
    predicted = {}
    for aid, path in enumerate(existing_paths):
        preds = []
        for dt in range(PRED+1):
            t = current_timestep + dt
            if t < len(path): preds.append(path[t])
            else: preds.append(path[-1] if path else agents[aid].initial_position)
        predicted[aid] = preds
    # detect conflicts
    conflict = set(failed_agents) | set(new_malfunction_agents)
    N = len(agents)
    for a in range(N):
        for b in range(a+1, N):
            for dt in range(PRED+1):
                pa, pb = predicted[a][dt], predicted[b][dt]
                if pa == pb or (dt>0 and pa == predicted[b][dt-1] and pb == predicted[a][dt-1]):
                    conflict |= {a, b}
                    break
    # reserve non-conflict
    for aid, path in enumerate(existing_paths):
        if aid not in conflict:
            for t in range(current_timestep, len(path)):
                reserved_cells.setdefault(t, set()).add(path[t])
                if t > current_timestep:
                    reserved_edges.setdefault(t, set()).add((path[t-1], path[t]))
    # dynamic obstacles
    for aid in new_malfunction_agents:
        pos = getattr(agents[aid], 'position', agents[aid].initial_position)
        for t in range(current_timestep, max_timestep+1):
            reserved_cells.setdefault(t, set()).add(pos)
    # replan conflict by priority
    for aid in sorted(conflict, key=lambda a: priority_rank[a]):
        agent = agents[aid]
        start_pos = getattr(agent, 'position', None) or agent.initial_position
        start_dir = getattr(agent, 'direction', agent.initial_direction)
        sub = time_space_astar(
            agent, current_timestep, start_pos, start_dir,
            rail, max_timestep, reserved_cells, reserved_edges
        )
        prefix = existing_paths[aid][:current_timestep]
        merged = prefix + sub
        existing_paths[aid] = merged
        for i, cell in enumerate(merged[current_timestep:]):
            t = current_timestep + i
            reserved_cells.setdefault(t, set()).add(cell)
            if i > 0:
                reserved_edges.setdefault(t, set()).add((merged[current_timestep+i-1], cell))
    # pad
    full = max_timestep + 1
    for aid, path in enumerate(existing_paths):
        if not path:
            existing_paths[aid] = [agents[aid].initial_position] * full
        elif len(path) < full:
            last = path[-1]
            existing_paths[aid] = path + [last] * (full - len(path))
    return existing_paths

if __name__ == "__main__":
    if len(sys.argv)>1:
        remote_evaluator(get_path, sys.argv, replan=replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path, "multi_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,
                                               f"multi_test_case/level{level}_test_{test}.pkl"))
        test_cases.sort()
        deadline_files = [tc.replace('.pkl','.ddl') for tc in test_cases]
        evaluator(get_path, test_cases, debug, visualizer, 3, deadline_files, replan=replan)
