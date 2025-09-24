from lib_piglet.utils.data_structure import bin_heap
from lib_piglet.search.search_node import search_node
from typing import Callable, Dict
import os


class item_handle:
    def __init__(self):
        self.focal_handle = None
        self.open_handle = None


"""
The default compare function for FOCAL.
You could modify it to observe the changes on FOCAL search behavior.

Sliding/analysis task: allow multiple secondary orders without changing call sites.
Control via environment variable FOCAL_SECONDARY_ORDER:
  - 'h_min'   : smaller h first (default)
  - 'h_max'   : larger h first
  - 'g_min'   : smaller g first
  - 'f_min'   : smaller f first
"""

FOCAL_SECONDARY_ORDER = os.getenv("FOCAL_SECONDARY_ORDER", "h_min").strip().lower()


def compare_function_2(a: search_node, b: search_node):
    # NOTE: bin_heap.compare_function should return True if a >= b. Using
    # comparisons below yields a min-heap for the chosen key when we use
    # the ">=" relation, and a max-heap if we use the "<=" relation.
    if FOCAL_SECONDARY_ORDER == "h_max":
        return a.h_ <= b.h_
    elif FOCAL_SECONDARY_ORDER == "g_min":
        return a.g_ >= b.g_
    elif FOCAL_SECONDARY_ORDER == "f_min":
        return a.f_ >= b.f_
    # default: 'h_min' — smaller h first
    return a.h_ >= b.h_


"""
This data structure maintains two priority queue, the focal list and open list. 
It will track the value of f_min and w*f_min and decide where a node will be stroed.
"""


class focal_priority_queue:

    def __init__(
        self,
        compare_function_1: Callable,
        compare_function_2=compare_function_2,
        weight: float = 1.0,
    ):
        self.weight: float = weight  # The suboptimality weight
        self.focal: bin_heap = bin_heap(compare_function_2)
        self.open: bin_heap = bin_heap(compare_function_1)
        self.f_min = 0
        self.w_f_min = 0
        self.id = 0  # The id of the next item inserted to FOCAL or OPEN

        # For the item with an id, use this dictionary to store the corresponding handle of the item in FOCAL or in OPEN
        # We need to use handle to tell bin_heap who to increase/decrease
        self.handles: Dict[int, item_handle] = {}

    def push(self, item: search_node) -> int:
        id = self.id
        self.id += 1

        # Create the corresponding object to store the open handle and focal handle for the item
        # When you push a node to OPEN/FOCAL you should record the returned handle to open_handle/focal_handle to this object
        # So that in increase/decrease function, we know what handle to pass to OPEN/FOCAL for increase/decrease
        self.handles[id] = item_handle()

        # The id of the item will also be recored in the search node,
        # so that when you pop a node from OPEN/FOCAL you can access the id and
        # use it to find corresponding item_handle object in self.handles
        item.priority_queue_handle_ = id

        ###########
        # Implement your code to handle how a search_noded is pushed to FOCAL/OPEN
        ###########

        # All nodes go to OPEN first (sorted by f1 value)
        open_handle = self.open.push(item)
        self.handles[id].open_handle = open_handle

        # Initialize f_min and w_f_min if this is the first node
        if self.f_min == 0:
            self.f_min = item.f_
            self.w_f_min = self.weight * self.f_min

        # Q4: Sliding window requires nodes in [f_min, w*f_min] to be in FOCAL
        # if item.f_ <= self.w_f_min:
        if self.f_min <= item.f_ <= self.w_f_min:
            focal_handle = self.focal.push(item)
            self.handles[id].focal_handle = focal_handle

        return id

    def pop(self) -> search_node:
        """
        Pop and return the top node from FOCAL.
        Don't forget to remove the corresponding heap_handle from the FOCAL.
        """
        ###########
        # Implement your code to handle how a search_noded is poped from FOCAL/OPEN
        # In a Conservative Window Stragety, before you pop a node, you would like to
        # check if FOCAL is empty, if yes, you should update the f_min, w_f_min and
        # bring nodes from OPEN to FOCAL.
        ###########

        # Q4
        # if self.focal.size() == 0:
        self.update_focal()

        # Pop from FOCAL (should not be empty after update_focal)
        if self.focal.size() == 0:
            return None  # No more nodes to process

        node = self.focal.pop()

        # Get the node's ID and handle
        node_id = node.priority_queue_handle_
        handle = self.handles[node_id]

        # Remove from OPEN as well (node is being expanded)
        if handle.open_handle is not None:
            self.open.erase(handle.open_handle)

        # Clean up handles
        handle.focal_handle = None
        handle.open_handle = None
        del self.handles[node_id]

        return node

    def update_focal(self):
        """
        This function is used to update the f_min and w_f_min.
        Then it will bring nodes in OPEN with f_ <= w_f_min to FOCAL
        Remember to update the open_handle/focal_handle when you move node from one to another
        """
        # Q4: Update when FOCAL is empty OR when OPEN.top().f_ increases
        if len(self.open) == 0:
            return

        ###########
        # Implement your code here. This function will update f_min and w_f_min and bring nodes from OPEN to FOCAL.
        # Use this function whenever you think is approciate.
        ###########

        # Determine whether to slide the window
        open_top = self.open.top()
        should_slide = (len(self.focal) == 0) or (self.f_min < open_top.f_)
        if should_slide:
            self.f_min = open_top.f_
            self.w_f_min = self.weight * self.f_min

        # Ensure all nodes with f in [f_min, w*f_min] are in FOCAL
        for i in range(1, self.open.currentSize + 1):
            heap_item = self.open.heapList[i]
            node = heap_item.item_
            if self.f_min <= node.f_ <= self.w_f_min:
                handle = self.handles[node.priority_queue_handle_]
                if handle.focal_handle is None:
                    focal_handle = self.focal.push(node)
                    handle.focal_handle = focal_handle

    def decrease(self, handle: int):
        """
        The f1 value of a node decreased, update the OPEN.
        Check if Focal need to be updated according to f2.
        """
        h = self.handles[handle]
        if h.focal_handle != None:
            self.focal.auto_update(h.focal_handle)
        if h.open_handle != None:
            ###########
            # Implement your code here,
            # Check if the f_ value is decreased below w_f_min, if yes and the node is not in focal, it should be pushed to focal.
            # Call decrease for open if the node is still in the OPEN.
            ###########

            # Get the node from OPEN
            node = self.open.get(h.open_handle)

            # Update the node's position in OPEN (since its f-value decreased)
            self.open.decrease(h.open_handle)

            # Q4: Check if the node now falls into [f_min, w*f_min]
            # if self.w_f_min > 0 and node.f_ <= self.w_f_min and h.focal_handle is None:  
            if self.w_f_min > 0 and self.f_min <= node.f_ <= self.w_f_min and h.focal_handle is None:
                focal_handle = self.focal.push(node)
                h.focal_handle = focal_handle

    def clear(self):
        self.open.clear()
        self.focal.clear()
        self.handles.clear()
        self.f_min = 0
        self.w_f_min = 0
        self.id = 0

    def __len__(self):
        return self.focal.size() + self.open.size()


