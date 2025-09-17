from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
from lib_piglet.utils.tools import eprint
import glob, os, sys, time, json
import heapq
import copy

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

# Set these debug1 option to True if you want more information printed
debug = False
debug1 = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
test_single_level = False
level = 1
test = 7
# 0,6
#########################
# Reimplementing the content in get_path() function and replan() function.
#
# They both return a list of paths. A path is a list of (x,y) location tuples.
# The path should be conflict free.
# Hint, you could use some global variables to reuse many resources across get_path/replan frunction calls.
#########################


class ReservationTable:
    """Reservation table that stores vertex and edge reservations."""

    def __init__(self) -> None:
        self.max_timestep: int = 0
        self.vertex = defaultdict(dict)
        self.edge = defaultdict(dict)

    def reset(self, max_timestep: int) -> None:
        self.max_timestep = int(max_timestep)
        self.vertex.clear()
        self.edge.clear()

    def reserve_vertex(self, agent_id: int, position, timestep: int) -> None:
        if timestep > self.max_timestep:
            return
        self.vertex[timestep][position] = agent_id

    def reserve_edge(
        self, agent_id: int, start, end, timestep: int
    ) -> None:
        if timestep > self.max_timestep:
            return
        self.edge[timestep][(start, end)] = agent_id

    def is_conflict(
        self, agent_id: int, current, nxt, next_timestep: int
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
        self, agent_id, path, upto_timestep
    ):
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
        self, agent_id, path, start_timestep
    ):
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
planned_paths = []
agent_order = []
max_timestep_global = 0

# Removed STALL_THRESHOLD logic for simplicity


# ========== Tools functions ==========
def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""

    if pos1 is None or pos2 is None:
        return 0
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def successors(rail: GridTransitionMap, loc, direction: int):
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


def compute_safe_intervals(agent_id: int, pos):
    """Return safe time intervals for a vertex excluding reservations by the same agent."""

    occupied = []
    for t, pos_dict in reservation_table.vertex.items():
        if pos_dict.get(pos) is not None and pos_dict[pos] != agent_id:
            occupied.append(t)
    occupied.sort()
    intervals = []
    start = 0
    for t in occupied:
        if t - 1 >= start:
            intervals.append((start, t - 1))
        start = t + 1
    if start <= reservation_table.max_timestep:
        intervals.append((start, reservation_table.max_timestep))
    return intervals


def form_new_path(old_path, new_path, split_t):
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


# Direction-aware delay toggle
DELAY_ONLY_SAME_MOVEMENT = False  # True: 仅顺延同一进出方向的后继；False: 同 cell 全顺延


def dir_from(a, b):
    if a is None or b is None:
        return None
    x1, y1 = a
    x2, y2 = b
    if x2 == x1 - 1 and y2 == y1:
        return 0  # 北
    if x2 == x1 and y2 == y1 + 1:
        return 1  # 东
    if x2 == x1 + 1 and y2 == y1:
        return 2  # 南
    if x2 == x1 and y2 == y1 - 1:
        return 3  # 西
    return None


def step_dirs(path, t):
    # t 时刻处于 path[t]；入方向由 t-1->t，出方向由 t->t+1
    in_dir = None if t - 1 < 0 else dir_from(path[t - 1], path[t])
    out_dir = None if t + 1 >= len(path) else dir_from(path[t], path[t + 1])
    return in_dir, out_dir


def compute_additional_waits(agents, paths, current_timestep, malfunction_agents):
    # 为每个 agent 计算需要统一延后的额外等待步数（取该 agent 所有未来 cell 访问中所需延后量的最大值）
    extra_wait = defaultdict(int)

    # 构建未来 cell 访问队列：pos -> [(orig_t, aid, in_dir, out_dir)]
    cell_queue = defaultdict(list)
    for aid, path in enumerate(paths):
        if not path:
            continue
        # 仅考虑未来时刻
        end_t = min(len(path), max_timestep_global + 1)
        for t in range(current_timestep + 1, end_t):
            pos = path[t]
            if pos is None:
                continue
            in_dir, out_dir = step_dirs(path, t)
            cell_queue[pos].append((t, aid, in_dir, out_dir))

    if debug1:
        print(f"[CELL] t={current_timestep} build queues: {len(cell_queue)} cells; malfunction_agents={list(malfunction_agents)}")

    # 对每个 cell 处理顺延
    for pos, events in cell_queue.items():
        if not events:
            continue
        # 按原计划到达时间排序
        events.sort(key=lambda x: x[0])

        # 分组（按进出方向）或整体处理
        if DELAY_ONLY_SAME_MOVEMENT:
            groups = defaultdict(list)
            for (t0, aid, in_d, out_d) in events:
                groups[(in_d, out_d)].append((t0, aid, in_d, out_d))
            group_list = groups.values()
        else:
            group_list = [events]

        for group in group_list:
            # 先拷贝原时刻，按 malfunction agent 对自身事件施加 base 延迟，再链式保证严格递增（至少 +1）
            prev_new_t = None
            for idx, (t0, aid, in_d, out_d) in enumerate(group):
                # 基础延迟：若该事件主体是 malfunction agent，则延 d
                base_delay = 0
                if aid in malfunction_agents:
                    try:
                        d = malfunction_remaining(agents[aid])
                    except Exception:
                        d = 0
                    # 仅作用于未来事件（t0 > current_timestep），条件已满足
                    base_delay = max(0, d)

                new_t = t0 + base_delay
                if prev_new_t is not None and new_t <= prev_new_t:
                    new_t = prev_new_t + 1
                prev_new_t = new_t

                # 记录该 agent 需要的最大统一延后量
                extra = max(0, new_t - t0)
                if extra > extra_wait[aid]:
                    extra_wait[aid] = extra

                if debug1 and (base_delay > 0 or extra > 0):
                    print(f"[CELL] pos={pos} aid={aid} t0={t0} base_delay={base_delay} new_t={new_t} extra={extra}")

    if debug1 and extra_wait:
        print(f"[CELL] extra_waits: {dict(extra_wait)}")
    return extra_wait


def compute_ingress_waits(agents, paths, current_timestep, malfunction_agents):
    # 对进入“故障占用cell”的后继进行窗口内入块检测：
    # 若某 agent 在 [t_cur+1, t_cur+d] 期间计划到达故障cell，则令其额外等待到 (t_cur+d+1)
    waits = defaultdict(int)
    if not malfunction_agents:
        return waits
    for m in malfunction_agents:
        if m < 0 or m >= len(agents):
            continue
        try:
            d = malfunction_remaining(agents[m])
        except Exception:
            d = 0
        if d <= 0:
            continue
        m_pos, _ = get_current_position(agents[m])
        if m_pos is None:
            continue
        window_end = min(current_timestep + d, max_timestep_global)
        for aid, path in enumerate(paths):
            if aid == m or not path:
                continue
            end_t = min(len(path) - 1, window_end)
            for t in range(current_timestep + 1, end_t + 1):
                if path[t] == m_pos:
                    need = (window_end + 1) - t
                    if need > waits[aid]:
                        waits[aid] = need
                    break
    if debug1 and waits:
        print(f"[INGRESS] waits: {dict(waits)}")
    return waits


def in_bounds(rail, pos):
    if pos is None:
        return False
    x, y = pos
    return 0 <= x < rail.height and 0 <= y < rail.width


def reconstruct_path(
    parents,
    goal_state,
    start_location,
    start_timestep,
    max_timestep,
):
    result = [start_location for _ in range(max_timestep + 1)]
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

    sequence = []
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
    start_location,
    start_direction: int,
    goal,
    start_timestep: int,
    max_timestep: int,
):
    if goal is None or start_location is None:
        return [start_location for _ in range(max_timestep + 1)]
    if start_location == goal:
        return [start_location for _ in range(max_timestep + 1)]

    start_intervals = compute_safe_intervals(agent_id, start_location)
    start_int = None
    for s, e in start_intervals:
        if s <= start_timestep <= e:
            start_int = (s, e)
            break
    if start_int is None:
        return [start_location for _ in range(max_timestep + 1)]

    open_heap = []
    g_best = {}
    parents = {}

    start_state = (
        start_location[0],
        start_location[1],
        start_direction,
        start_timestep,
    )
    parents[start_state] = None
    g_best[
        (
            start_location[0],
            start_location[1],
            start_direction,
            start_int[0],
            start_int[1],
        )
    ] = start_timestep

    # Use Manhattan distance as heuristic
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

    best_goal = None

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
                if rail.is_dead_end((nxt[0], nxt[1])) and reservation_table.vertex.get(arrival, {}).get((nxt[0], nxt[1])) is not None and (nxt[0], nxt[1]) != goal:
                    continue
                nkey = (nxt[0], nxt[1], ndir, ns, ne)
                if arrival >= g_best.get(nkey, float("inf")):
                    continue
                g_best[nkey] = arrival
                child_state = (nxt[0], nxt[1], ndir, arrival)
                parents[child_state] = (x, y, direction, g)
                # Use Manhattan distance as heuristic
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

    return reconstruct_path(
        parents, best_goal, start_location, start_timestep, max_timestep
    )


def get_path(agents, rail, max_timestep):
    """Plan initial paths for all agents with prioritized planning."""

    global planned_paths, agent_order, max_timestep_global
    max_timestep_global = int(max_timestep)
    reservation_table.reset(max_timestep_global)

    planned_paths = [[] for _ in range(len(agents))]

    # priorities disabled: use default agent order instead of sorting by slack/ddl/distance
    priorities = []  # (agent_id, slack, ddl_value, est_distance, agent_id)
    for agent_id, agent in enumerate(agents):
        # if agent_id == 4:
        #     print(f'agent 4 direction: {agent.direction} {agent.initial_direction}')
        start = agent.initial_position
        goal = agent.target if agent.target is not None else start
        ddl = getattr(agent, "deadline", None)
        est_distance = manhattan_distance(start, goal)
        ddl_value = ddl if ddl is not None else max_timestep_global
        slack = ddl_value - est_distance
        priorities.append((agent_id, slack, ddl_value, est_distance, agent_id))
    agent_order = [
        item[0] for item in sorted(priorities, key=lambda x: (x[1], x[2], x[3]))
    ]
    # agent_order = list(range(len(agents)))
    # print(f'[GET_PATH] agent_order: {agent_order}')
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
    agents,
    rail,
    current_timestep,
    existing_paths,
    max_timestep,
    new_malfunction_agents,
    failed_agents,
):
    """Replan paths for agents affected by malfunctions or failures."""

    global planned_paths, agent_order, max_timestep_global
    max_timestep_global = int(max_timestep)
    planned_paths = existing_paths

    reservation_table.reset(max_timestep_global)

    current_timestep = int(current_timestep)

    # Reserve history for all agents up to the current timestep
    for agent_id, path in enumerate(planned_paths):
        if not path:
            continue
        reservation_table.reserve_history(agent_id, path, current_timestep)

    # 计算“当前仍在故障”的集合，用于等待与顺延计算（而不是仅仅依赖新发生的故障）
    current_malfunction_set = set(i for i, a in enumerate(agents) if malfunction_remaining(a) > 0)

    # 基于同 cell 顺延与入块检测的静态等待（以当前仍在故障的集合为依据）
    extra_waits = compute_additional_waits(agents, planned_paths, current_timestep, current_malfunction_set) if current_malfunction_set else {}
    ingress_waits = compute_ingress_waits(agents, planned_paths, current_timestep, current_malfunction_set) if current_malfunction_set else {}
    combined_waits = defaultdict(int)
    for k, v in extra_waits.items():
        combined_waits[k] = max(combined_waits[k], v)
    for k, v in ingress_waits.items():
        combined_waits[k] = max(combined_waits[k], v)

    # 待重规划集合：新增或失败 + 当前仍在故障 + 被顺延/入块影响到需要等待的
    new_or_failed_set = set(new_malfunction_agents) | set(failed_agents)
    to_replan = set(a for a, w in combined_waits.items() if w > 0) | new_or_failed_set | current_malfunction_set
    #if to_replan has agent 4, then add agent 3 to to_replan
    # if 7 in to_replan:
    #     print(f"[REPLAN] replan agent 7, ")
    # Keep reservations for agents not being replanned
    for agent_id, path in enumerate(planned_paths):
        if agent_id in to_replan or not path:
            continue
        reservation_table.reserve_path(agent_id, path, start_timestep=current_timestep)

    # priorities disabled: use default agent order instead of sorting by slack/ddl/distance
    priorities = []
    for agent_id, agent in enumerate(agents):
        loc, _ = get_current_position(agent)
        goal = agent.target if agent.target is not None else loc
        ddl = getattr(agent, "deadline", None)
        est_distance = manhattan_distance(loc, goal)
        ddl_value = ddl if ddl is not None else max_timestep_global
        slack = ddl_value - est_distance
        priorities.append((agent_id, slack, ddl_value, est_distance, agent_id))
    agent_order = [
        item[0] for item in sorted(priorities, key=lambda x: ( x[2],))
    ]
    # print(f"[REPLAN] agent_order: {agent_order}")
    # agent_order = list(range(len(agents)))

    replan_sequence = [k for k in agent_order if k in to_replan]

    if debug1:
        print(f"[REPLAN] t={current_timestep} to_replan={sorted(list(to_replan))} waits={dict(combined_waits) if combined_waits else {}}")

    for agent_id in replan_sequence:
        if agent_id not in to_replan:
            continue
        agent = agents[agent_id]
        cur_loc, cur_dir = get_current_position(agent)
        wait_steps = 0
        # 1) 自身故障等待（以“当前仍在故障集”为准）
        if agent_id in current_malfunction_set:
            wait_steps = max(wait_steps, malfunction_remaining(agent))
        # 2) 基于 cell 排队的统一延后
        if agent_id in combined_waits:
            wait_steps = max(wait_steps, combined_waits.get(agent_id, 0))

        for offset in range(wait_steps):
            t = current_timestep + offset
            reservation_table.reserve_vertex(agent_id, cur_loc, t)
            if t > 0:
                reservation_table.reserve_edge(agent_id, cur_loc, cur_loc, t - 1)

        start_time = current_timestep + wait_steps
        if debug1:
            print(f"[REPLAN] aid={agent_id} wait_steps={wait_steps} start_time={start_time} cur_loc={cur_loc}")
        new_path = single_agent_sipp(
            agent_id,
            rail,
            cur_loc,
            cur_dir,
            agent.target,
            start_timestep=start_time,
            max_timestep=max_timestep_global,
        )
        # if agent_id == 7:
        #     print(f"[REPLAN] new_path: {new_path}")
        merged_path = form_new_path(
            planned_paths[agent_id], new_path, split_t=current_timestep
        )
        planned_paths[agent_id] = merged_path
        # 立即写预约，使后续重规划的 agent 能看到最新约束
        reservation_table.reserve_path(
            agent_id, merged_path, start_timestep=current_timestep
        )

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
