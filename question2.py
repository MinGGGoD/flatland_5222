"""
This is the python script for question 2. In this script, you are required to implement a multi-agent path-finding algorithm
"""

from lib_piglet.utils.tools import eprint
import glob, os, sys
import heapq

# import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
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
    eprint("Cannot load flatland modules!", e)
    exit(1)

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = True
visualizer = True

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = True
level = 1
test = 0

#########################
# Reimplementing the content in get_path() function.
#
# Return a list of (x,y) location tuples which connect the start and goal locations.
# The path should avoid conflicts with existing paths.
#########################

# (dx, dy) offsets
offsets = {
    Directions.NORTH: (-1, 0),
    Directions.EAST: (0, 1),
    Directions.SOUTH: (1, 0),
    Directions.WEST: (0, -1),
}


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def check_bounds(height, width, x, y) -> bool:
    return 0 <= x < height and 0 <= y < width


def build_reservations(existing_paths: list, max_timestep: int):
    """
    Build reservation tables from other agents' fixed paths.
    Parameters:
        existing_paths: list of lists of tuples, each tuple is a (x, y) coordinate
        max_timestep: the maximum time step
    Returns:
        reserve_vertices: set of (x, y, t)
        reserve_edges: set of ((x1,y1),(x2,y2),t)
    """
    reserve_vertices = set()
    reserve_edges = set()
    for p in existing_paths:
        if not p:
            continue
        # Reserve each occupied vertex at its timestep
        for t in range(min(len(p), max_timestep + 1)):
            x, y = p[t]
            reserve_vertices.add((x, y, t))  # vertex occupied at time t

        # Reserve edges between continuous positions to prevent head-on swaps
        for t in range(min(len(p) - 1, max_timestep)):
            x1, y1 = p[t]
            x2, y2 = p[t + 1]
            reserve_edges.add(
                ((x1, y1), (x2, y2), t)
            )  # other train at time t goes from (x1,y1) to (x2,y2)

        # Park at goal after arrival: keep last cell occupied up to max_timestep
        last = p[-1]
        for t in range(len(p), max_timestep + 1):
            reserve_vertices.add(
                (last[0], last[1], t)
            ) 
    return reserve_vertices, reserve_edges


def successors(rail: GridTransitionMap, x: int, y: int, d: int):
    """
    Generate actions successors.
    """
    successor_list = []
    
    # WAIT
    successor_list.append((x, y, d, "WAIT"))

    valid_actions = rail.get_transitions(x, y, d) # 返回上右下左是否允许移动
    # print(valid_actions) # 4 elements tuple (0, 0, 0, 1)
    for next_dir, allowed in enumerate(valid_actions):
        # print('next_dir, allowed =>', next_dir, allowed)
        if not allowed:
            continue
        dx, dy = offsets[next_dir]
        nx, ny = x + dx, y + dy
        if check_bounds(rail.height, rail.width, nx, ny):
            successor_list.append((nx, ny, next_dir, "MOVE"))
    
    return successor_list


def edge_conflict(u: tuple, v: tuple, t: int, reserve_edges: set) -> bool:
    """
    Edge swap / traversal conflict check.
    """
    return ((u, v, t) in reserve_edges) or ((v, u, t) in reserve_edges)


def reconstruct_path(parents: dict, end_state: tuple):
    """
    Reconstruct positions per timestep from (x,y,dir,t).
    """
    path_rev = []
    s = end_state
    while s in parents:
        x, y, d, t = s
        path_rev.append((x, y))
        s = parents[s]
    # add the very first state's position
    x0, y0, d0, t0 = s
    path_rev.append((x0, y0))
    path_rev.reverse()
    return path_rev


def is_start_permanently_occupied(start, reserve_vertices, max_timestep):
    """
    Check if start position is permanently occupied
    """
    for t in range(max_timestep + 1):
        if (start[0], start[1], t) in reserve_vertices:
            # If occupied from t=0 to max_timestep, it's permanently occupied
            continue
        else:
            # If not occupied at some time point, it's not permanently occupied
            return False
    return True



# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param agent_id The id of given agent
# @param existing_paths A list of lists of locations indicate existing paths. The index of each location is the time that
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path(
    start: tuple,
    start_direction: int,
    goal: tuple,
    rail: GridTransitionMap,
    agent_id: int,
    existing_paths: list,
    max_timestep: int,
):
    """
    Time-expanded A* with reservation table (single agent vs moving obstacles)
    """
    # if debug:
    #     print(f"Agent {agent_id} 起点: {start}, 起点方向: {start_direction}, 目标: {goal}")

    # Goal check
    if start == goal:
        return [start]

    # Generate reservation tables base on existing paths
    # max_timestep = int(max_timestep) 
    reserve_vertices, reserve_edges = build_reservations(existing_paths, max_timestep)

    # Statespace: (x, y, dir, timestep)
    start_state = (start[0], start[1], start_direction, 0)

    # Check initial vertex if occupied permanently
    if is_start_permanently_occupied(start, reserve_vertices, max_timestep):
        return []

    # OPEN set (priority queue) and bookkeeping
    open_heap = []
    tie = 0
    g0 = 0
    f0 = g0 + manhattan_distance((start[0], start[1]), goal)
    # Priority: (f_value, h_value, time, tie, state)
    heapq.heappush(open_heap, (f0, manhattan_distance((start[0], start[1]), goal), 0, tie, start_state))
    tie += 1

    # Pruning: record earliest arrival time for (x,y,dir)
    best_arrival = {}  # key=(x,y,dir)->t
    
    # Parent pointers for path reconstruct
    parents = {}

    # A*
    while open_heap:
        _, _, _, _, (x, y, d, t) = heapq.heappop(open_heap)

        # Goal test
        if (x, y) == goal:
            path = reconstruct_path(parents, (x, y, d, t))
            return path

        # Generate successors for next timestep
        for nx, ny, nd, move_type in successors(rail, x, y, d):
            t2 = t + 1
            if t2 > max_timestep: # time exceed
                continue

            # Vertex conflict
            if (nx, ny, t2) in reserve_vertices:
                continue

            # Edge conflict
            if not (nx == x and ny == y):
                if edge_conflict((x, y), (nx, ny), t, reserve_edges):
                    continue

            # Pruning
            key = (nx, ny, nd)
            prev_best = best_arrival.get(key, None)
            if prev_best is not None and prev_best <= t2:
                continue
            best_arrival[key] = t2

            # Costs: g = time so far, h = Manhattan distance
            g2 = t2
            f2 = g2 + manhattan_distance((nx, ny), goal)

            # Update open set
            s2 = (nx, ny, nd, t2)
            parents[s2] = (x, y, d, t)
            heapq.heappush(open_heap, (f2, manhattan_distance((nx, ny), goal), t2, tie, s2))
            tie += 1

    # No feasible path 
    return []


#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path, sys.argv)
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
        test_cases.sort()
        evaluator(get_path, test_cases, debug, visualizer, 2)
