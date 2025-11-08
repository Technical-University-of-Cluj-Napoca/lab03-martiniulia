from utils import *
from collections import deque
from queue import PriorityQueue
from grid import Grid
from spot import Spot
import math

def bfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    queue = deque()
    queue.append(start)
    visited = {start}
    came_from = {} 
    while queue:
        current = queue.popleft()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()
    return False

def dfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    stack = [start]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event == pygame.QUIT :
                pygame.quit()
        ##if quit: pygame.quit()
        ##draw()
        current = stack.pop()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end(), start.make_start()
            return True
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()

    return False

def dls (draw: callable, grid: Grid, start: Spot, end: Spot, limit : int) -> bool:
    stack = [(start, 0)]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event == pygame.QUIT :
                pygame.quit()
        ##if quit: pygame.quit()
        ##draw()
        current, depth = stack.pop()
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end(), start.make_start()
            return True
        if depth == limit:
            continue
        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append((neighbor, depth + 1))
                neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()

    return False

def h_manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Manhattan distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    x1, y1 = p1.get_position()
    x2, y2 = p2.get_position()
    return abs(x1 - x2) + abs(y1 - y2)
    

def h_euclidian_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Euclidian distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    x1, y1 = p1.get_position()
    x2, y2 = p2.get_position()
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def astar(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    count = 0
    open_heap = PriorityQueue()
    open_heap.put((0, count, start)) 

    came_from = {}
    g_score = {spot: float("inf") for row in grid.grid for spot in row}
    f_score = {spot: float("inf") for row in grid.grid for spot in row}

    g_score[start] = 0 
    f_score[start] = h_manhattan_distance(start, end)

    lookup_set = {start}

    while not open_heap.empty():
        current = open_heap.get()[2]
        lookup_set.remove(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            start.make_start()
            end.make_end()
            return True
        
        for neigh in current.neighbors:
            tentative_g = g_score[current] + 1
            if tentative_g < g_score[neigh]:
                came_from[neigh] = current
                g_score[neigh] = tentative_g
                f_score[neigh] = tentative_g + h_manhattan_distance(neigh, end)
                if neigh not in lookup_set:
                    count += 1
                    open_heap.put((f_score[neigh], count, neigh))
                    lookup_set.add(neigh)
                    neigh.make_open()
        draw()

        if current != start:
            current.make_closed()
    
    return False

def ucs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    count = 0
    open_heap = PriorityQueue()
    open_heap.put((0, count, start)) 

    came_from = {}
    g_score = {spot: float("inf") for row in grid.grid for spot in row}
    g_score[start] = 0 

    lookup_set = {start}

    while not open_heap.empty():
        current = open_heap.get()[2]
        lookup_set.remove(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            start.make_start()
            end.make_end()
            return True
        
        for neigh in current.neighbors:
            tentative_g = g_score[current] + 1
            if tentative_g < g_score[neigh]:
                came_from[neigh] = current
                g_score[neigh] = tentative_g
                if neigh not in lookup_set:
                    count += 1
                    open_heap.put((g_score[neigh], count, neigh))
                    lookup_set.add(neigh)
                    neigh.make_open()
        draw()

        if current != start:
            current.make_closed()
    
    return False

def greedy(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    count = 0
    open_heap = PriorityQueue()
    open_heap.put((0, count, start)) 

    came_from = {}
    visited = set()

    while not open_heap.empty():
        current = open_heap.get()[2]
        visited.add(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            start.make_start()
            end.make_end()
            return True
        
        for neigh in current.neighbors:
            if neigh not in visited and not neigh.is_barrier():
                came_from[neigh] = current
                priority = h_manhattan_distance(neigh, end)
                count += 1
                open_heap.put((priority, count, neigh))
                neigh.make_open()
        draw()

        if current != start:
            current.make_closed()
    
    return False

def ids(draw: callable, grid: Grid, start : Spot, end: Spot, max_depth: int) -> bool:
    for depth in range(0, max_depth):
        for row in grid.grid:
            for spot in row:
                if not spot.is_barrier() and spot != start and spot != end:
                    spot.reset()
        if dls(draw, grid, start, end, depth):
            return True
    return 

def helper(draw: callable, grid: Grid,start: Spot, end: Spot, threshold: float) -> tuple[bool, float]:
    stack = [(start, 0)]
    came_from = {}
    min_th = float('inf')

    while stack:
        current, g = stack.pop()
        f = g + h_manhattan_distance(current, end)
        if f > threshold:
            min_th = min(min_th, f)
            continue
        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            start.make_start()
            end.make_end()
            return True, f
        for neighbor in current.neighbors:
            if neighbor not in came_from and not neighbor.is_barrier():
                came_from[neighbor] = current
                stack.append((neighbor, g + 1))
                neighbor.make_open()
        draw()
        if current != start:
            current.make_closed()
    return False, min_th

def ida(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    threshold = h_manhattan_distance(start, end)
    while True:
        found, min_th = helper(draw, grid, start, end, threshold)
        if found:
            return True
        if min_th == float('inf'):
            return False
        threshold = min_th
          
            
# and the others algorithms...
# ▢ Depth-Limited Search (DLS)
# ▢ Uninformed Cost Search (UCS)
# ▢ Greedy Search
# ▢ Iterative Deepening Search/Iterative Deepening Depth-First Search (IDS/IDDFS)
# ▢ Iterative Deepening A* (IDA)
# Assume that each edge (graph weight) equalss