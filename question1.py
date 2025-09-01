"""
This is the python script for question 1. In this script, you are required to implement a single agent path-finding algorithm
"""
from lib_piglet.utils.tools import eprint
import glob, os, sys

#import necessary modules that this python scripts need.
try:
    from flatland.core.transition_map import GridTransitionMap
    from flatland.utils.controller import get_action, Train_Actions, Directions, check_conflict, path_controller, evaluator, remote_evaluator
except Exception as e:
    eprint("Cannot load flatland modules!")
    eprint(e)
    exit(1)

import heapq

#########################
# Debugger and visualizer options
#########################

# Set these debug option to True if you want more information printed
debug = False
visualizer = False

# If you want to test on specific instance, turn test_single_instance to True and specify the level and test number
test_single_instance = False
level = 1
test = 0


#########################
# Reimplementing the content in get_path() function.
#
# Return a list of (x,y) location tuples which connect the start and goal locations.
#########################

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_next_position(x, y, direction):
    if direction == Directions.NORTH:
        return (x - 1, y)
    elif direction == Directions.EAST:
        return (x, y + 1)
    elif direction == Directions.SOUTH:
        return (x + 1, y)
    elif direction == Directions.WEST:
        return (x, y - 1)
    else:
        return (x, y)
    

# This function return a list of location tuple as the solution.
# @param start A tuple of (x,y) coordinates
# @param start_direction An Int indicate direction.
# @param goal A tuple of (x,y) coordinates
# @param rail The flatland railway GridTransitionMap
# @param max_timestep The max timestep of this episode.
# @return path A list of (x,y) tuple.
def get_path1(start: tuple, start_direction: int, goal: tuple, rail: GridTransitionMap, max_timestep: int):
    """ A* algorithm implementation """

    open_set = []
    heapq.heappush(open_set, (0, 0, start, start_direction, [start]))
    visited = set()
    g_score = {} # g_value set
    g_score[start, start_direction] = 0
    
    while open_set:
    
        # pop best node
        f_score, current_g, current_pos, current_direction, current_path = heapq.heappop(open_set)
        
        # check time limit
        if len(current_path) >= max_timestep:
            continue
        
        current_state = (current_pos, current_direction)
        # check duplicate visits
        if current_state in visited:
            continue
        visited.add(current_state)
        
        # goal-test
        if current_pos == goal:
            return current_path
        
        # get valid actions
        valid_actions = rail.get_transitions(current_pos[0], current_pos[1], current_direction)
        
        # iterate all possible actions and calculate the new state
        for new_direction in range(len(valid_actions)):
        
            if valid_actions[new_direction]:
                # new position
                new_pos = get_next_position(current_pos[0], current_pos[1], new_direction)
                
                # check boundary
                if new_pos[0] < 0 or new_pos[0] >= rail.width or new_pos[1] < 0 or new_pos[1] >= rail.height:
                    continue
                
                # new state
                new_state = (new_pos, new_direction)

                # check duplicate visits
                if new_state in visited:
                    continue
                
                # calculate new g with cost=1
                new_g = current_g + 1
                
                # compare new_g and current g
                if new_state not in g_score or new_g < g_score[new_state]:
                    # update g value
                    g_score[new_state] = new_g
                    
                    # heuristic cost
                    h_score = manhattan_distance(new_pos, goal)
                    
                    # calculate f value
                    f_score = new_g + h_score
                    
                    # append path
                    new_path = current_path + [new_pos]
                    
                    # update queue
                    heapq.heappush(open_set, (f_score, new_g, new_pos, new_direction, new_path))
    
    return None



#########################
# You should not modify codes below, unless you want to modify test_cases to test specific instance. You can read it know how we ran flatland environment.
########################
if __name__ == "__main__":
    if len(sys.argv) > 1:
        remote_evaluator(get_path,sys.argv)
    else:
        script_path = os.path.dirname(os.path.abspath(__file__))
        test_cases = glob.glob(os.path.join(script_path,"single_test_case/level*_test_*.pkl"))
        if test_single_instance:
            test_cases = glob.glob(os.path.join(script_path,"single_test_case/level{}_test_{}.pkl".format(level, test)))
        test_cases.sort()
        evaluator(get_path,test_cases,debug,visualizer,1)



















