"""
Graph Search Algorithms: BFS, DFS, A*
For discrete problems like path-finding
"""

import numpy as np
from collections import deque
import heapq


class GridWorld:
    """
    Grid World environment for path-finding
    
    Parameters:
    -----------
    grid_size : tuple
        (rows, cols) size of grid
    obstacles : list of tuples, optional
        List of (row, col) obstacle positions
    start : tuple, optional
        (row, col) start position
    goal : tuple, optional
        (row, col) goal position
    obstacle_prob : float, optional
        Probability of obstacle at each cell (for random generation)
    seed : int, optional
        Random seed
    """
    
    def __init__(self, grid_size=(20, 20), obstacles=None, start=None, 
                 goal=None, obstacle_prob=0.2, seed=None):
        self.rows, self.cols = grid_size
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize grid (0 = free, 1 = obstacle)
        self.grid = np.zeros(grid_size, dtype=int)
        
        if obstacles is not None:
            for r, c in obstacles:
                self.grid[r, c] = 1
        else:
            # Generate random obstacles
            for i in range(self.rows):
                for j in range(self.cols):
                    if np.random.rand() < obstacle_prob:
                        self.grid[i, j] = 1
        
        # Set start and goal
        if start is None:
            self.start = (0, 0)
        else:
            self.start = start
        
        if goal is None:
            self.goal = (self.rows - 1, self.cols - 1)
        else:
            self.goal = goal
        
        # Ensure start and goal are not obstacles
        self.grid[self.start] = 0
        self.grid[self.goal] = 0
    
    def is_valid(self, pos):
        """Check if position is valid (in bounds and not obstacle)"""
        r, c = pos
        return (0 <= r < self.rows and 0 <= c < self.cols and 
                self.grid[r, c] == 0)
    
    def get_neighbors(self, pos):
        """Get valid neighbors of position (4-connected)"""
        r, c = pos
        neighbors = []
        
        # Up, Right, Down, Left
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            new_pos = (r + dr, c + dc)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def get_neighbors_8(self, pos):
        """Get valid neighbors of position (8-connected)"""
        r, c = pos
        neighbors = []
        
        # 8 directions
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_pos = (r + dr, c + dc)
                if self.is_valid(new_pos):
                    neighbors.append(new_pos)
        
        return neighbors
    
    def manhattan_distance(self, pos1, pos2):
        """Manhattan distance heuristic"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def euclidean_distance(self, pos1, pos2):
        """Euclidean distance heuristic"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def summary(self):
        """Print problem summary"""
        n_obstacles = np.sum(self.grid)
        print(f"Grid World Summary:")
        print(f"  Grid size: {self.rows} x {self.cols}")
        print(f"  Start: {self.start}")
        print(f"  Goal: {self.goal}")
        print(f"  Obstacles: {n_obstacles} ({n_obstacles / (self.rows * self.cols) * 100:.1f}%)")


class BreadthFirstSearch:
    """
    Breadth-First Search (BFS) Algorithm
    Guarantees shortest path in unweighted graphs
    """
    
    @staticmethod
    def search(grid_world, verbose=False):
        """
        Run BFS to find path from start to goal
        
        Parameters:
        -----------
        grid_world : GridWorld
            Grid world instance
        verbose : bool
            Print progress
        
        Returns:
        --------
        path : list of tuples
            Path from start to goal (or None if no path exists)
        path_length : int
            Length of path
        nodes_expanded : int
            Number of nodes expanded during search
        """
        start = grid_world.start
        goal = grid_world.goal
        
        # Queue: (position)
        queue = deque([start])
        
        # Track visited nodes and parent pointers
        visited = {start}
        parent = {start: None}
        
        nodes_expanded = 0
        
        while queue:
            current = queue.popleft()
            nodes_expanded += 1
            
            if verbose and nodes_expanded % 100 == 0:
                print(f"Expanded {nodes_expanded} nodes, queue size: {len(queue)}")
            
            # Check if goal reached
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                
                if verbose:
                    print(f"Path found! Length: {len(path)}, Nodes expanded: {nodes_expanded}")
                
                return path, len(path), nodes_expanded
            
            # Explore neighbors
            for neighbor in grid_world.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        
        # No path found
        if verbose:
            print(f"No path found. Nodes expanded: {nodes_expanded}")
        
        return None, 0, nodes_expanded


class DepthFirstSearch:
    """
    Depth-First Search (DFS) Algorithm
    Does not guarantee shortest path
    """
    
    @staticmethod
    def search(grid_world, max_depth=None, verbose=False):
        """
        Run DFS to find path from start to goal
        
        Parameters:
        -----------
        grid_world : GridWorld
            Grid world instance
        max_depth : int, optional
            Maximum depth to explore
        verbose : bool
            Print progress
        
        Returns:
        --------
        path : list of tuples
            Path from start to goal (or None if no path exists)
        path_length : int
            Length of path
        nodes_expanded : int
            Number of nodes expanded during search
        """
        start = grid_world.start
        goal = grid_world.goal
        
        if max_depth is None:
            max_depth = grid_world.rows * grid_world.cols
        
        # Stack: (position, depth)
        stack = [(start, 0)]
        
        # Track visited nodes and parent pointers
        visited = {start}
        parent = {start: None}
        
        nodes_expanded = 0
        
        while stack:
            current, depth = stack.pop()
            nodes_expanded += 1
            
            if verbose and nodes_expanded % 100 == 0:
                print(f"Expanded {nodes_expanded} nodes, stack size: {len(stack)}")
            
            # Check if goal reached
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                
                if verbose:
                    print(f"Path found! Length: {len(path)}, Nodes expanded: {nodes_expanded}")
                
                return path, len(path), nodes_expanded
            
            # Explore neighbors (if not too deep)
            if depth < max_depth:
                for neighbor in grid_world.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        parent[neighbor] = current
                        stack.append((neighbor, depth + 1))
        
        # No path found
        if verbose:
            print(f"No path found. Nodes expanded: {nodes_expanded}")
        
        return None, 0, nodes_expanded


class AStarSearch:
    """
    A* Search Algorithm
    Guarantees optimal path with admissible heuristic
    """
    
    @staticmethod
    def search(grid_world, heuristic='manhattan', verbose=False):
        """
        Run A* to find path from start to goal
        
        Parameters:
        -----------
        grid_world : GridWorld
            Grid world instance
        heuristic : str
            Heuristic function ('manhattan' or 'euclidean')
        verbose : bool
            Print progress
        
        Returns:
        --------
        path : list of tuples
            Path from start to goal (or None if no path exists)
        path_length : int
            Length of path
        nodes_expanded : int
            Number of nodes expanded during search
        """
        start = grid_world.start
        goal = grid_world.goal
        
        # Choose heuristic
        if heuristic == 'manhattan':
            h = lambda pos: grid_world.manhattan_distance(pos, goal)
        else:
            h = lambda pos: grid_world.euclidean_distance(pos, goal)
        
        # Priority queue: (f_score, counter, position)
        # counter is used to break ties
        counter = 0
        open_set = [(h(start), counter, start)]
        
        # Track visited nodes
        visited = set()
        
        # Track g_score (cost from start)
        g_score = {start: 0}
        
        # Track parent pointers
        parent = {start: None}
        
        nodes_expanded = 0
        
        while open_set:
            # Get node with lowest f_score
            f, _, current = heapq.heappop(open_set)
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            nodes_expanded += 1
            
            if verbose and nodes_expanded % 100 == 0:
                print(f"Expanded {nodes_expanded} nodes, open set size: {len(open_set)}")
            
            # Check if goal reached
            if current == goal:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                
                if verbose:
                    print(f"Path found! Length: {len(path)}, Nodes expanded: {nodes_expanded}")
                    print(f"Path cost: {g_score[goal]}")
                
                return path, len(path), nodes_expanded
            
            # Explore neighbors
            for neighbor in grid_world.get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + 1
                
                # Update if better path found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + h(neighbor)
                    parent[neighbor] = current
                    
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))
        
        # No path found
        if verbose:
            print(f"No path found. Nodes expanded: {nodes_expanded}")
        
        return None, 0, nodes_expanded


class SearchAlgorithmsComparison:
    """Compare BFS, DFS, and A* on same problem"""
    
    @staticmethod
    def compare(grid_world, verbose=False):
        """
        Compare all three algorithms
        
        Returns:
        --------
        results : dict
            Dictionary with results for each algorithm
        """
        results = {}
        
        # BFS
        if verbose:
            print("\n" + "="*50)
            print("Running BFS...")
            print("="*50)
        path, length, expanded = BreadthFirstSearch.search(grid_world, verbose=verbose)
        results['BFS'] = {
            'path': path,
            'path_length': length,
            'nodes_expanded': expanded,
            'found': path is not None
        }
        
        # DFS
        if verbose:
            print("\n" + "="*50)
            print("Running DFS...")
            print("="*50)
        path, length, expanded = DepthFirstSearch.search(grid_world, verbose=verbose)
        results['DFS'] = {
            'path': path,
            'path_length': length,
            'nodes_expanded': expanded,
            'found': path is not None
        }
        
        # A* (Manhattan)
        if verbose:
            print("\n" + "="*50)
            print("Running A* (Manhattan)...")
            print("="*50)
        path, length, expanded = AStarSearch.search(grid_world, heuristic='manhattan', verbose=verbose)
        results['A* (Manhattan)'] = {
            'path': path,
            'path_length': length,
            'nodes_expanded': expanded,
            'found': path is not None
        }
        
        # A* (Euclidean)
        if verbose:
            print("\n" + "="*50)
            print("Running A* (Euclidean)...")
            print("="*50)
        path, length, expanded = AStarSearch.search(grid_world, heuristic='euclidean', verbose=verbose)
        results['A* (Euclidean)'] = {
            'path': path,
            'path_length': length,
            'nodes_expanded': expanded,
            'found': path is not None
        }
        
        return results
    
    @staticmethod
    def print_comparison(results):
        """Print comparison table"""
        print("\n" + "="*70)
        print("SEARCH ALGORITHMS COMPARISON")
        print("="*70)
        print(f"{'Algorithm':<20} {'Path Found':<12} {'Path Length':<12} {'Nodes Expanded':<15}")
        print("-"*70)
        
        for algo_name, result in results.items():
            found = "Yes" if result['found'] else "No"
            length = result['path_length'] if result['found'] else "-"
            expanded = result['nodes_expanded']
            print(f"{algo_name:<20} {found:<12} {str(length):<12} {expanded:<15}")
        
        print("="*70)

