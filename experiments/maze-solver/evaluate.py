"""
Evaluation harness for A* maze solver heuristic.
Generates 20 reproducible 50x50 mazes and measures total nodes expanded.
Output format: SCORE: <total_nodes_expanded>
"""
import heapq
import random
import sys
import importlib.util

GRID_SIZE = 50
NUM_MAZES = 20
OBSTACLE_RATIO = 0.25
SEED = 42

def generate_mazes():
    rng = random.Random(SEED)
    mazes = []
    for _ in range(NUM_MAZES):
        while True:
            grid = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    if (r, c) == (0, 0) or (r, c) == (GRID_SIZE-1, GRID_SIZE-1):
                        continue
                    if rng.random() < OBSTACLE_RATIO:
                        grid[r][c] = 1
            if bfs_path_exists(grid, (0,0), (GRID_SIZE-1, GRID_SIZE-1)):
                mazes.append(grid)
                break
    return mazes

def bfs_path_exists(grid, start, goal):
    from collections import deque
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr,nc) not in visited and grid[nr][nc] == 0:
                visited.add((nr,nc))
                queue.append((nr,nc))
    return False

def bfs_true_distances(grid, goal):
    """Compute true shortest distances from all reachable cells to goal."""
    from collections import deque
    dist = {}
    queue = deque([goal])
    dist[goal] = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr][nc] == 0 and (nr,nc) not in dist:
                dist[(nr,nc)] = dist[(r,c)] + 1
                queue.append((nr,nc))
    return dist

def astar(grid, heuristic_fn):
    start = (0, 0)
    goal = (GRID_SIZE-1, GRID_SIZE-1)
    counter = 0
    open_set = [(heuristic_fn(0, 0, goal[0], goal[1], grid, GRID_SIZE), counter, 0, 0)]
    g_score = {start: 0}
    closed_set = set()
    nodes_expanded = 0
    while open_set:
        f, _, r, c = heapq.heappop(open_set)
        if (r, c) in closed_set:
            continue
        closed_set.add((r, c))
        nodes_expanded += 1
        if (r, c) == goal:
            return nodes_expanded
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr][nc] == 0 and (nr, nc) not in closed_set:
                new_g = g_score[(r,c)] + 1
                if new_g < g_score.get((nr,nc), float('inf')):
                    g_score[(nr,nc)] = new_g
                    h = heuristic_fn(nr, nc, goal[0], goal[1], grid, GRID_SIZE)
                    counter += 1
                    heapq.heappush(open_set, (new_g + h, counter, nr, nc))
    return nodes_expanded

def check_admissibility(heuristic_fn, mazes):
    goal = (GRID_SIZE-1, GRID_SIZE-1)
    for i, grid in enumerate(mazes[:5]):
        true_dist = bfs_true_distances(grid, goal)
        for (r, c), dist in true_dist.items():
            h = heuristic_fn(r, c, goal[0], goal[1], grid, GRID_SIZE)
            if h > dist + 1e-9:
                print(f"FAIL: Not admissible - maze {i}: h({r},{c})={h:.4f} > true={dist}")
                return False
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <solution.py>")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"FAIL: Could not load module: {e}")
        sys.exit(1)

    if not hasattr(mod, 'heuristic'):
        print("FAIL: Module must define a 'heuristic' function")
        sys.exit(1)

    heuristic_fn = mod.heuristic
    mazes = generate_mazes()

    if not check_admissibility(heuristic_fn, mazes):
        sys.exit(1)

    total = 0
    for maze in mazes:
        total += astar(maze, heuristic_fn)

    print(f"SCORE: {total}")

if __name__ == "__main__":
    main()
