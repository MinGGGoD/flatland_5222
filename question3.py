from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from lib_piglet.utils.tools import eprint
import glob, os, sys, time, json
import heapq

# import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.envs.agent_utils import EnvAgent
    from flatland.utils.controller import (
        get_action,
        Train_Actions,
        Directions,
        check_conflict,
        path_controller,
        evaluator,
        remote_evaluator,
    )
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
test = 5
# 0,6
#########################
# Reimplementing the content in get_path() function and replan() function.
#
# They both return a list of paths. A path is a list of (x,y) location tuples.
# The path should be conflict free.
# Hint, you could use some global variables to reuse many resources across get_path/replan frunction calls.
#########################


Position = Tuple[int, int]
State = Tuple[int, int, int, int]


class ReservationTable:
    """Reservation table that stores vertex and edge reservations."""

    def __init__(self) -> None:
        self.max_timestep: int = 0
        self.vertex: Dict[int, Dict[Position, int]] = defaultdict(dict)
        self.edge: Dict[int, Dict[Tuple[Position, Position], int]] = defaultdict(dict)

    def reset(self, max_timestep: int) -> None:
        self.max_timestep = int(max_timestep)
        self.vertex.clear()
        self.edge.clear()

    def reserve_vertex(self, agent_id: int, position: Position, timestep: int) -> None:
        if timestep > self.max_timestep:
            return
        self.vertex[timestep][position] = agent_id

    def reserve_edge(
        self, agent_id: int, start: Position, end: Position, timestep: int
    ) -> None:
        if timestep > self.max_timestep:
            return
        self.edge[timestep][(start, end)] = agent_id

    def is_conflict(
        self, agent_id: int, current: Position, nxt: Position, next_timestep: int
    ) -> bool:
        if next_timestep > self.max_timestep:
            return True
        occupied = self.vertex.get(next_timestep, {}).get(nxt)
        if occupied is not None and occupied != agent_id:
            return True
        if next_timestep > 0:
            reverse = self.edge.get(next_timestep - 1, {}).get((nxt, current))
            if reverse is not None and reverse != agent_id:
                return True
        return False

    def reserve_history(
        self, agent_id: int, path: List[Position], upto_timestep: int
    ) -> None:
        if not path:
            return
        end = min(upto_timestep, len(path) - 1, self.max_timestep)
        for t in range(end):
            cur = path[t]
            nxt = path[t + 1]
            self.reserve_vertex(agent_id, cur, t)
            self.reserve_vertex(agent_id, nxt, t + 1)
            self.reserve_edge(agent_id, cur, nxt, t)

    def reserve_path(
        self, agent_id: int, path: List[Position], start_timestep: int
    ) -> None:
        if not path:
            return
        end_time = min(len(path) - 1, self.max_timestep)
        for t in range(start_timestep, end_time):
            cur = path[t]
            nxt = path[t + 1]
            self.reserve_vertex(agent_id, cur, t)
            self.reserve_vertex(agent_id, nxt, t + 1)
            self.reserve_edge(agent_id, cur, nxt, t)
        last_pos = path[end_time]
        for t in range(max(start_timestep, end_time), self.max_timestep + 1):
            self.reserve_vertex(agent_id, last_pos, t)
            if t > 0:
                self.reserve_edge(agent_id, last_pos, last_pos, t - 1)


reservation_table = ReservationTable()
planned_paths: List[List[Position]] = []
agent_order: List[int] = []
max_timestep_global = 0


# ========== Tools functions ==========
def manhattan_distance(pos1: Optional[Position], pos2: Optional[Position]) -> int:
    """Calculate Manhattan distance between two positions."""

    if pos1 is None or pos2 is None:
        return 0
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def successors(rail: GridTransitionMap, loc: Position, direction: int):
    """Generate successor states including waiting."""

    x, y = loc
    neighbors = []
    valid = rail.get_transitions(x, y, direction)
    for d in range(4):
        if valid[d]:
            nx, ny = x, y
            if d == 0:
                nx -= 1
            elif d == 1:
                ny += 1
            elif d == 2:
                nx += 1
            elif d == 3:
                ny -= 1
            neighbors.append(((nx, ny), d))
    neighbors.append(((x, y), direction))
    return neighbors


def compute_safe_intervals(agent_id: int, pos: Position) -> List[Tuple[int, int]]:
    """Return safe time intervals for a vertex excluding reservations by the same agent."""

    occupied = []
    for t, pos_dict in reservation_table.vertex.items():
        if pos_dict.get(pos) is not None and pos_dict[pos] != agent_id:
            occupied.append(t)
    occupied.sort()
    intervals: List[Tuple[int, int]] = []
    start = 0
    for t in occupied:
        if t - 1 >= start:
            intervals.append((start, t - 1))
        start = t + 1
    if start <= reservation_table.max_timestep:
        intervals.append((start, reservation_table.max_timestep))
    return intervals


def form_new_path(
    old_path: List[Position], new_path: List[Position], split_t: int
) -> List[Position]:
    """Merge an updated suffix with the preserved prefix of an old plan."""

    if split_t <= 0 or not old_path:
        return new_path
    prefix = old_path[:split_t]
    suffix = new_path[split_t:]
    return prefix + suffix


def get_current_position(agent: EnvAgent):
    """Get the agent current pose."""

    loc = getattr(agent, "position", None)
    if loc is None:
        loc = agent.initial_position
    d = getattr(agent, "direction", None)
    if d is None:
        d = agent.initial_direction
    return loc, d


def malfunction_remaining(agent: EnvAgent) -> int:
    """Get remaining malfunction timesteps."""

    data = getattr(agent, "malfunction_data", None)
    if data and "malfunction" in data:
        return int(data["malfunction"])
    return 0


def in_bounds(rail: GridTransitionMap, pos: Optional[Position]) -> bool:
    if pos is None:
        return False
    x, y = pos
    return 0 <= x < rail.height and 0 <= y < rail.width


from collections import deque


distance_map_cache: Dict[Tuple[int, int], List[List[int]]] = {}


def get_distance_map(rail: GridTransitionMap, goal: Optional[Position]):
    global distance_map_cache
    if goal is None or not in_bounds(rail, goal):
        return [[10**9] * rail.width for _ in range(rail.height)]

    key = (goal[0], goal[1])
    if key in distance_map_cache:
        return distance_map_cache[key]

    dist_map = [[10**9] * rail.width for _ in range(rail.height)]
    q = deque([goal])
    dist_map[goal[0]][goal[1]] = 0

    while q:
        x, y = q.popleft()
        base_cost = dist_map[x][y]
        for direction in range(4):
            transitions = rail.get_transitions(x, y, direction)
            for move_dir in range(4):
                if not transitions[move_dir]:
                    continue
                nx, ny = x, y
                if move_dir == 0:
                    nx -= 1
                elif move_dir == 1:
                    ny += 1
                elif move_dir == 2:
                    nx += 1
                elif move_dir == 3:
                    ny -= 1
                if not (0 <= nx < rail.height and 0 <= ny < rail.width):
                    continue
                if dist_map[nx][ny] > base_cost + 1:
                    dist_map[nx][ny] = base_cost + 1
                    q.append((nx, ny))

    distance_map_cache[key] = dist_map
    return dist_map


def reconstruct_path(
    parents: Dict[State, Optional[State]],
    goal_state: Optional[State],
    start_location: Position,
    start_timestep: int,
    max_timestep: int,
) -> List[Position]:
    result: List[Position] = [start_location for _ in range(max_timestep + 1)]
    visited = [False for _ in range(max_timestep + 1)]
    if 0 <= start_timestep <= max_timestep:
        result[start_timestep] = start_location
        visited[start_timestep] = True

    if goal_state is None:
        for t in range(start_timestep - 1, -1, -1):
            result[t] = start_location
        for t in range(start_timestep + 1, max_timestep + 1):
            result[t] = start_location
        return result

    sequence: List[State] = []
    cursor = goal_state
    while cursor is not None:
        sequence.append(cursor)
        cursor = parents[cursor]
    sequence.reverse()

    for x, y, direction, timestep in sequence:
        if 0 <= timestep <= max_timestep:
            result[timestep] = (x, y)
            visited[timestep] = True

    last_pos = start_location
    for t in range(start_timestep, max_timestep + 1):
        if not visited[t]:
            result[t] = last_pos
        else:
            last_pos = result[t]

    for t in range(start_timestep - 1, -1, -1):
        result[t] = start_location

    return result


def single_agent_sipp(
    agent_id: int,
    rail: GridTransitionMap,
    start_location: Position,
    start_direction: int,
    goal: Optional[Position],
    start_timestep: int,
    max_timestep: int,
) -> List[Position]:
    if goal is None or start_location is None:
        return [start_location for _ in range(max_timestep + 1)]
    if start_location == goal:
        return [start_location for _ in range(max_timestep + 1)]

    dist_map = get_distance_map(rail, goal)

    start_intervals = compute_safe_intervals(agent_id, start_location)
    start_int = None
    for s, e in start_intervals:
        if s <= start_timestep <= e:
            start_int = (s, e)
            break
    if start_int is None:
        return [start_location for _ in range(max_timestep + 1)]

    open_heap: List[Tuple[int, int, int, int, int, int, int, int]] = []
    g_best: Dict[Tuple[int, int, int, int, int], int] = {}
    parents: Dict[State, Optional[State]] = {}

    start_state = (start_location[0], start_location[1], start_direction, start_timestep)
    parents[start_state] = None
    g_best[(start_location[0], start_location[1], start_direction, start_int[0], start_int[1])] = start_timestep

    h0 = dist_map[start_location[0]][start_location[1]]
    if h0 >= 10**9:
        h0 = manhattan_distance(start_location, goal)

    heapq.heappush(
        open_heap,
        (
            start_timestep + h0,
            start_timestep,
            start_location[0],
            start_location[1],
            start_direction,
            start_int[0],
            start_int[1],
        ),
    )

    best_goal: Optional[State] = None

    while open_heap:
        f, g, x, y, direction, int_s, int_e = heapq.heappop(open_heap)
        key = (x, y, direction, int_s, int_e)
        if g != g_best.get(key, float("inf")):
            continue
        if (x, y) == goal:
            best_goal = (x, y, direction, g)
            break
        if g >= max_timestep:
            continue

        for nxt, ndir in successors(rail, (x, y), direction):
            if not in_bounds(rail, nxt):
                continue
            intervals = compute_safe_intervals(agent_id, nxt)
            for ns, ne in intervals:
                depart_time = max(g, ns - 1)
                if depart_time > int_e:
                    continue
                arrival = depart_time + 1
                if arrival < ns or arrival > ne:
                    continue
                if reservation_table.is_conflict(agent_id, (x, y), nxt, arrival):
                    continue
                nkey = (nxt[0], nxt[1], ndir, ns, ne)
                if arrival >= g_best.get(nkey, float("inf")):
                    continue
                g_best[nkey] = arrival
                child_state: State = (nxt[0], nxt[1], ndir, arrival)
                parents[child_state] = (x, y, direction, g)
                h = dist_map[nxt[0]][nxt[1]]
                if h >= 10**9:
                    h = manhattan_distance(nxt, goal)
                heapq.heappush(
                    open_heap,
                    (
                        arrival + h,
                        arrival,
                        nxt[0],
                        nxt[1],
                        ndir,
                        ns,
                        ne,
                    ),
                )

    return reconstruct_path(parents, best_goal, start_location, start_timestep, max_timestep)


def get_path(agents: List[EnvAgent], rail: GridTransitionMap, max_timestep: int):
    """Plan initial paths for all agents with prioritized planning."""

    global distance_map_cache, planned_paths, agent_order, max_timestep_global
    distance_map_cache = {}
    max_timestep_global = int(max_timestep)
    reservation_table.reset(max_timestep_global)

    planned_paths = [[] for _ in range(len(agents))]

    priorities = []  # (agent_id, slack, ddl_value, est_distance, agent_id)
    for agent_id, agent in enumerate(agents):
        start = agent.initial_position
        goal = agent.target if agent.target is not None else start
        ddl = getattr(agent, "deadline", None)
        est_distance = manhattan_distance(start, goal)
        ddl_value = ddl if ddl is not None else max_timestep_global
        slack = ddl_value - est_distance
        priorities.append((agent_id, slack, ddl_value, est_distance, agent_id))

    agent_order = [item[0] for item in sorted(priorities, key=lambda x: (x[1], x[2], x[3], x[4]))]

    for agent_id in agent_order:
        agent = agents[agent_id]
        path = single_agent_sipp(
            agent_id,
            rail,
            agent.initial_position,
            agent.initial_direction,
            agent.target,
            start_timestep=0,
            max_timestep=max_timestep_global,
        )
        planned_paths[agent_id] = path
        reservation_table.reserve_path(agent_id, path, start_timestep=0)

    return planned_paths


def replan(
    agents: List[EnvAgent],
    rail: GridTransitionMap,
    current_timestep: int,
    existing_paths,
    max_timestep: int,
    new_malfunction_agents,
    failed_agents,
):
    """Replan paths for agents affected by malfunctions or failures."""

    global planned_paths, agent_order, max_timestep_global, distance_map_cache
    max_timestep_global = int(max_timestep)
    planned_paths = existing_paths
    distance_map_cache = {}
    reservation_table.reset(max_timestep_global)

    current_timestep = int(current_timestep)

    # Reserve history for all agents up to the current timestep
    for agent_id, path in enumerate(planned_paths):
        if not path:
            continue
        reservation_table.reserve_history(agent_id, path, current_timestep)

    to_replan = set(new_malfunction_agents) | set(failed_agents)

    # Keep reservations for agents not being replanned
    for agent_id, path in enumerate(planned_paths):
        if agent_id in to_replan or not path:
            continue
        reservation_table.reserve_path(agent_id, path, start_timestep=current_timestep)

    priorities = []
    for agent_id, agent in enumerate(agents):
        loc, _ = get_current_position(agent)
        goal = agent.target if agent.target is not None else loc
        ddl = getattr(agent, "deadline", None)
        est_distance = manhattan_distance(loc, goal)
        ddl_value = ddl if ddl is not None else max_timestep_global
        slack = ddl_value - est_distance
        priorities.append((agent_id, slack, ddl_value, est_distance, agent_id))

    agent_order = [item[0] for item in sorted(priorities, key=lambda x: (x[1], x[2], x[3], x[4]))]

    for agent_id in agent_order:
        if agent_id not in to_replan:
            continue
        agent = agents[agent_id]
        cur_loc, cur_dir = get_current_position(agent)
        wait_steps = malfunction_remaining(agent) if agent_id in new_malfunction_agents else 0

        for offset in range(wait_steps):
            t = current_timestep + offset
            reservation_table.reserve_vertex(agent_id, cur_loc, t)
            if t > 0:
                reservation_table.reserve_edge(agent_id, cur_loc, cur_loc, t - 1)

        start_time = current_timestep + wait_steps
        new_path = single_agent_sipp(
            agent_id,
            rail,
            cur_loc,
            cur_dir,
            agent.target,
            start_timestep=start_time,
            max_timestep=max_timestep_global,
        )
        merged_path = form_new_path(planned_paths[agent_id], new_path, split_t=current_timestep)
        planned_paths[agent_id] = merged_path
        reservation_table.reserve_path(agent_id, merged_path, start_timestep=current_timestep)

    return planned_paths


#####################################################################
# Instantiate a Remote Client
# You should not modify codes below, unless you want to modify test_cases to test specific instance.
#####################################################################
if __name__ == "__main__":

    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv, replan=replan)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(
            os.path.join(script_path, "multi_test_case/level*_test_*.pkl")
        )

        if test_single_instance:
            test_cases = glob.glob(
                os.path.join(
                    script_path,
                    "multi_test_case/level{}_test_{}.pkl".format(level, test),
                )
            )
        elif test_single_level:
            test_cases = glob.glob(
                os.path.join(script_path, f"multi_test_case/level{level}_test_*.pkl")
            )
        test_cases.sort()
        deadline_files = [test.replace(".pkl", ".ddl") for test in test_cases]
        evaluator(
            get_path, test_cases, debug, visualizer, 3, deadline_files, replan=replan
        )
