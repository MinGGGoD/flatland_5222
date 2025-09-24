# solution/solution.py
# This module defines a base search class and what attribute/method a search class should have.
#
# @author: mike
# @created: 2020-07-16
#

# Store solution of a search. paths_ attribute include a list of search_node.
class solution:

    # Get solution statistic
    # @return list List of solution cost and depth.
    def get_solution_info(self):
        # return [self.cost_, self.depth_]
    # Format cost to fixed width of 9 characters total
        if isinstance(self.cost_, (int, float)):
            # Convert to string with enough precision
            cost_str = f"{self.cost_:.7f}"
            # Truncate or pad to exactly 9 characters
            if len(cost_str) > 9:
                formatted_cost = cost_str[:9]
            else:
                formatted_cost = cost_str.ljust(9)
        else:
            formatted_cost = str(self.cost_).ljust(9)
        return [formatted_cost, self.depth_]

    def __init__(self, path: list, depth: int, cost: float):
        self.cost_: float = cost
        self.depth_: int = depth
        self.paths_: list = path

    def __str__(self):
        return "{}".format(self.paths_)

    def __repr__(self):
        return self.__str__()

# Convert solution to a list of state
def solution_to_state_list(sol: solution):
    path = solution.paths_[:]
    for i in range(0,len(path)):
        path = path[i].state_
    return path

