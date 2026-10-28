import heapq

# =============================================================================
# 1. A* Pathfinding Search
# =============================================================================
grid = [
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_list = [(heuristic(start, goal), 0, start, [start])]
    visited = set()
    while open_list:
        f, g, node, path = heapq.heappop(open_list)
        if node == goal:
            return path, g
        if node in visited:
            continue
        visited.add(node)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = node[0] + dr, node[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                nxt = (nr, nc)
                if nxt not in visited:
                    g2 = g + 1
                    f2 = g2 + heuristic(nxt, goal)
                    heapq.heappush(open_list, (f2, g2, nxt, path + [nxt]))
    return None, None

# =============================================================================
# 2. Alpha-Beta Pruning Algorithm
# =============================================================================
pruned_count = 0

def alpha_beta(node, depth, alpha, beta, maximizing):
    global pruned_count
    if not isinstance(node, list):
        return node
    if maximizing:
        best = float("-inf")
        for i, child in enumerate(node):
            val = alpha_beta(child, depth + 1, alpha, beta, False)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                pruned_count += len(node) - i - 1
                print(f"  Pruned {len(node) - i - 1} branch(es) at depth {depth} (MAX)")
                break
        return best
    else:
        best = float("inf")
        for i, child in enumerate(node):
            val = alpha_beta(child, depth + 1, alpha, beta, True)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                pruned_count += len(node) - i - 1
                print(f"  Pruned {len(node) - i - 1} branch(es) at depth {depth} (MIN)")
                break
        return best

# =============================================================================
# Execution
# =============================================================================
if __name__ == "__main__":
    print("=== A* SEARCH ===")
    path, cost = a_star(grid, (0, 0), (4, 4))
    print("Path :", path)
    print("Cost :", cost)

    print("\n=== ALPHA-BETA PRUNING ===")
    tree = [[[3, 5], [6, 9]], [[1, 2], [0, -1]]]
    result = alpha_beta(tree, 0, float("-inf"), float("inf"), True)
    print("Best value for MAX player:", result)
    print("Total branches pruned :", pruned_count)