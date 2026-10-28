from collections import deque
from itertools import permutations
import heapq

# =============================================================================
# 1. N-Queen Problem (Backtracking / Depth-First Search)
# =============================================================================
def is_safe(board, row, col):
    for r in range(row):
        c = board[r]
        if c == col or abs(c - col) == abs(r - row):
            return False
    return True

def solve_nqueen(n, row=0, board=None):
    if board is None:
        board = [-1] * n
    if row == n:
        return board[:]
    for col in range(n):
        if is_safe(board, row, col):
            board[row] = col
            result = solve_nqueen(n, row + 1, board)
            if result:
                return result
            board[row] = -1
    return None

def print_nqueen(n):
    sol = solve_nqueen(n)
    print(f"N-Queen solution for N = {n}:")
    for r in range(n):
        print(" ".join("Q" if sol[r] == c else "." for c in range(n)))

# =============================================================================
# 2. Water Jug Problem (Breadth-First Search)
# =============================================================================
def water_jug(cap_a, cap_b, target):
    start = (0, 0)
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        (a, b), path = queue.popleft()
        if a == target or b == target:
            return path
        moves = [
            (cap_a, b), (a, cap_b),
            (0, b), (a, 0),
            (a - min(a, cap_b - b), b + min(a, cap_b - b)),
            (a + min(b, cap_a - a), b - min(b, cap_a - a))
        ]
        for m in moves:
            if m not in visited:
                visited.add(m)
                queue.append((m, path + [m]))
    return None

# =============================================================================
# 3. 8-Puzzle Problem (Best-First Search / Misplaced Tiles Heuristic)
# =============================================================================
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)

def misplaced(state):
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != GOAL[i])

def neighbours(state):
    i = state.index(0)
    r, c = divmod(i, 3)
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            s = list(state)
            s[i], s[j] = s[j], s[i]
            result.append(tuple(s))
    return result

def solve_puzzle(start):
    pq = [(misplaced(start), start, [start])]
    visited = {start}
    while pq:
        h, state, path = heapq.heappop(pq)
        if state == GOAL:
            return path
        for nb in neighbours(state):
            if nb not in visited:
                visited.add(nb)
                heapq.heappush(pq, (misplaced(nb), nb, path + [nb]))
    return None

def show_state(s):
    for i in range(0, 9, 3):
        print(" ".join(str(x) if x else "_" for x in s[i:i + 3]))
    print()

# =============================================================================
# 4. Traveling Salesperson Problem (Brute-force Permutations)
# =============================================================================
def tsp(dist):
    n = len(dist)
    best_cost, best_path = float("inf"), None
    for perm in permutations(range(1, n)):
        path = (0,) + perm + (0,)
        cost = sum(dist[path[i]][path[i + 1]] for i in range(n))
        if cost < best_cost:
            best_cost, best_path = cost, path
    return best_path, best_cost

# =============================================================================
# Execution
# =============================================================================
if __name__ == "__main__":
    print("=== 1. N-QUEEN ===")
    print_nqueen(6)

    print("\n=== 2. WATER JUG (4L, 3L, target 2L) ===")
    for step in water_jug(4, 3, 2):
        print(step)

    print("\n=== 3. 8-PUZZLE ===")
    start = (1, 2, 3, 4, 0, 6, 7, 5, 8)
    path = solve_puzzle(start)
    print("Steps needed:", len(path) - 1)
    for s in path:
        show_state(s)

    print("=== 4. TSP ===")
    dist = [[0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]]
    p, c = tsp(dist)
    print("Best route:", p)
    print("Minimum cost:", c)