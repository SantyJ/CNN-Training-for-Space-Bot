import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import numpy as np
import math
import sys
import os
from scipy.spatial.distance import cdist
from datetime import datetime
import multiprocessing
import time
from heapq import heappush, heappop
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Start the timer
start_time = time.time()

os.environ['PYTHONUNBUFFERED'] = "1"

GRID_SIZE = 30  # Grid size

max_cores = 15

class SimulationState:
    def __init__(self):
        self.movements = 0
        self.blocked_cell_actions = 0
        self.space_rat_detector_actions = 0
        self.failed_simulations = 0
        self.actions = []

    def reset(self):
        self.movements = 0
        self.blocked_cell_actions = 0
        self.space_rat_detector_actions = 0
        self.failed_simulations = 0
        self.actions = []

# Define the neural network within the existing script
# class SpaceRatNN(nn.Module):
#     def __init__(self, grid_size, num_channels, hidden_size):
#         super(SpaceRatNN, self).__init__()
#         self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(32 * grid_size * grid_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, 1)  # Single output for predicted actions remaining
#         self.relu = nn.ReLU()
#         self.flatten = nn.Flatten()

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.flatten(x)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)  # No activation on output for regression
#         return x

class SpaceRatNN(nn.Module):
    def __init__(self, grid_size, num_channels, hidden_size):
        super(SpaceRatNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * grid_size * grid_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Single output for predicted actions remaining
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # No activation on output for regression
        return x

# Create empty grid layout
def create_grid_backup(size):
    return [[1 for _ in range(size)] for _ in range(size)]  # 1 represents 'blocked'

def create_grid(size):
    grid = [[1 for _ in range(size)] for _ in range(size)]  # 1 represents 'blocked'
    for i in range(size):
        grid[0][i] = 1  # Top edge
        grid[size - 1][i] = 1  # Bottom edge
        grid[i][0] = 1  # Left edge
        grid[i][size - 1] = 1  # Right edge
    return grid

# Find a random open cell in the grid for the space rat's initial position
def get_random_open_position(grid):
    open_positions = [(x, y) for x in range(1, GRID_SIZE - 1) for y in range(1, GRID_SIZE - 1) if grid[x][y] == 0]
    return random.choice(open_positions) if open_positions else None

# Pick random starting cell in grid
def start_with_initial_cell(grid):
    x, y = random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2)
    grid[x][y] = 0  # 0 represents 'open'
    return (x, y)

def get_candidates_for_opening(grid):
    candidates = []
    for row in range(1, GRID_SIZE - 1):  # Exclude top and bottom edges
        for col in range(1, GRID_SIZE - 1):  # Exclude left and right edges
            if grid[row][col] == 1:  # 'blocked' cells
                open_neighbors = sum(
                    grid[nr][nc] == 0  # 'open' cells
                    for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                )
                if open_neighbors == 1:
                    candidates.append((row, col))
    return candidates

# Open the candidate cells retrieved in above function iteratively
def open_cells(grid):
    while True:
        candidates = get_candidates_for_opening(grid)
        if not candidates:
            break
        cell_to_open = random.choice(candidates)
        grid[cell_to_open[0]][cell_to_open[1]] = 0  # Set to 'open'

def find_dead_ends(grid):
    dead_ends = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] == 0:  # 'open' cells
                open_neighbors = sum(
                    grid[nr][nc] == 0
                    for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE
                )
                if open_neighbors == 1:
                    dead_ends.append((row, col))
    return dead_ends

def expand_dead_ends(grid, dead_ends):
    selected_dead_ends = random.sample(dead_ends, len(dead_ends) // 2)
    for dead_end in selected_dead_ends:
        row, col = dead_end
        blocked_neighbors = [
            (nr, nc)
            for nr, nc in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr][nc] == 1  # 'blocked'
        ]
        if blocked_neighbors:
            neighbor_to_open = random.choice(blocked_neighbors)
            grid[neighbor_to_open[0]][neighbor_to_open[1]] = 0  # Set to 'open'

# Check if 60 percent of grid open before expanding dead ends
def check_sixty_percent(grid):
    TARGET_OPEN_PERCENTAGE = 0.6  # Target percentage of open cells
    open_count = sum(row.count(0) for row in grid)  # Count 'open' cells
    total_cells = GRID_SIZE * GRID_SIZE
    return open_count >= TARGET_OPEN_PERCENTAGE * total_cells

def block_edge_cells(grid):
    # Block all cells in the first and last row
    for col in range(GRID_SIZE):
        grid[0][col] = 1  # Block the first row
        grid[GRID_SIZE - 1][col] = 1  # Block the last row

    # Block all cells in the first and last column
    for row in range(GRID_SIZE):
        grid[row][0] = 1  # Block the first column
        grid[row][GRID_SIZE - 1] = 1  # Block the last column

# Wrapper function for creating grid layout
def create_space_vessel_layout():
    while True:
        grid = create_grid(GRID_SIZE)
        start_with_initial_cell(grid)
        open_cells(grid)

        dead_ends = find_dead_ends(grid)
        expand_dead_ends(grid, dead_ends)

        # Ensure all edge cells are blocked
        block_edge_cells(grid)

        if check_sixty_percent(grid):
            pass
        else:
            # print("60 percent not open, trying again")
            continue

        break
                
    return grid

def convert_grid_to_numpy(grid):
    numpy_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] == 1:  # 'blocked'
                numpy_grid[r, c] = 1
    return numpy_grid

def sense_blocked_neighbors(grid, position, state):
    # global BLOCKED_CELL_ACTIONS
    x, y = position
    directions = [(x-1, y), (x+1, y), (x, y-1), (x, y+1), 
                  (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)]
    count_blocked = 0
    for nx, ny in directions:
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 1:
            count_blocked += 1
    # BLOCKED_CELL_ACTIONS = BLOCKED_CELL_ACTIONS + 1
    return count_blocked

def possible_positions_after_sensing(grid, sensed_blocked_neighbors, eliminated_positions, state):
    """
    Update possible positions based on sensed blocked neighbors using probabilities.
    """
    state.blocked_cell_actions += 1
    possible_positions = []
    for row in range(1, GRID_SIZE - 1):
        for col in range(1, GRID_SIZE - 1):
            if grid[row][col] == 0 and (row, col) not in eliminated_positions:  # Consider only open cells not in eliminated positions
                temp_blocked_neighbors = sense_blocked_neighbors(grid, (row, col), state)
                if temp_blocked_neighbors == sensed_blocked_neighbors:
                    possible_positions.append((row, col))
    return possible_positions


def move_bot_bkp(grid, possible_positions, current_position, previous_position):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    random.shuffle(directions)
    open_directions = {d: 0 for d in directions}

    # Count open directions among possible positions
    for pos in possible_positions:
        x, y = pos
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 0:
                open_directions[(dx, dy)] += 1

    # Sort directions by the count of open directions, in descending order
    sorted_directions = sorted(open_directions.items(), key=lambda item: item[1], reverse=True)

    # Find the maximum open directions count
    max_count = sorted_directions[0][1]
    best_directions = [d for d, count in sorted_directions if count == max_count]

    # Step 1: Choose a random direction from the best directions
    chosen_direction = random.choice(best_directions)

    # Step 2: Check if this direction leads back to the previous position
    if previous_position:
        reverse_direction = (previous_position[0] - current_position[0], previous_position[1] - current_position[1])
        if chosen_direction == reverse_direction:
            # Step 3: Choose another best direction if available
            if len(best_directions) > 1:
                best_directions.remove(reverse_direction)
                chosen_direction = random.choice(best_directions)
            else:
                # Step 4: If only one best direction and it is the reverse direction,
                # choose it, otherwise select the next best direction
                found_alternative = False
                for direction, _ in sorted_directions:
                    if direction != reverse_direction:
                        chosen_direction = direction
                        found_alternative = True
                        break
                if not found_alternative:
                    # If no other option exists, stick with the reverse direction
                    chosen_direction = reverse_direction

    # Try the selected direction
    new_x, new_y = current_position[0] + chosen_direction[0], current_position[1] + chosen_direction[1]
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and grid[new_x, new_y] == 0:
        return (new_x, new_y), True, chosen_direction  # Successful move

    # If no valid move is found, return the current position
    return current_position, False, chosen_direction  # No valid move possible

def move_bot_v2(grid, possible_positions, current_position, previous_position, visited_positions):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    random.shuffle(directions)
    open_directions = {d: 0 for d in directions}

    # Count open directions among possible positions
    for pos in possible_positions:
        x, y = pos
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 0:
                open_directions[(dx, dy)] += 1

    # Sort directions by the count of open directions, in descending order
    sorted_directions = sorted(open_directions.items(), key=lambda item: item[1], reverse=True)

    # Filter out directions leading to already visited or blocked positions
    filtered_directions = []
    for direction, _ in sorted_directions:
        new_x, new_y = current_position[0] + direction[0], current_position[1] + direction[1]
        if (new_x, new_y) not in visited_positions and 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and grid[new_x, new_y] == 0:
            filtered_directions.append(direction)

    # If all directions are visited or blocked, reset filtered directions to all possible (to avoid being stuck)
    if not filtered_directions:
        filtered_directions = [d for d, _ in sorted_directions if 0 <= current_position[0] + d[0] < GRID_SIZE and 0 <= current_position[1] + d[1] < GRID_SIZE and grid[current_position[0] + d[0], current_position[1] + d[1]] == 0]

    # If still no directions, consider the current position as a dead-end (return current state without change)
    if not filtered_directions:
        return current_position, False, (0, 0)  # No valid move

    # Choose a random direction from the best available directions
    chosen_direction = random.choice(filtered_directions)

    # Step 2: Check if this direction leads back to the previous position
    if previous_position:
        reverse_direction = (previous_position[0] - current_position[0], previous_position[1] - current_position[1])
        if chosen_direction == reverse_direction and len(filtered_directions) > 1:
            filtered_directions.remove(reverse_direction)
            chosen_direction = random.choice(filtered_directions)

    # Try the selected direction
    new_x, new_y = current_position[0] + chosen_direction[0], current_position[1] + chosen_direction[1]
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and grid[new_x, new_y] == 0:
        visited_positions.add((new_x, new_y))  # Mark as visited
        return (new_x, new_y), True, chosen_direction  # Successful move

    # If no valid move is found, return the current position
    return current_position, False, chosen_direction  # No valid move possible

def move_bot_bkp_latest(grid, possible_positions, current_position, previous_position, visited_positions):
    """
    DFS movement strategy that prioritizes open directions based on possible_positions
    but does not restrict movement solely to these positions. If no valid move is found among
    unvisited cells, it attempts a DFS-based move among visited cells (excluding previous_position).
    """
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    random.shuffle(directions)  # Shuffle directions to add variability

    # Calculate direction priorities based on open paths around possible_positions
    open_directions = {d: 0 for d in directions}
    for pos in possible_positions:
        x, y = pos
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx][ny] == 0:
                open_directions[(dx, dy)] += 1

    # Sort directions by the count of open directions, in descending order
    sorted_directions = sorted(open_directions.items(), key=lambda item: item[1], reverse=True)

    # DFS Stack Initialization
    stack = [(current_position, [])]
    visited = visited_positions.copy()  # Maintain a copy of the visited set
    visited.add(current_position)  # Mark current position as visited

    # Step 1: Attempt DFS exploration among unvisited cells
    while stack:
        pos, path = stack.pop()

        # Attempt to move in each sorted direction (prioritizing open paths)
        for (dx, dy), _ in sorted_directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            next_pos = (new_x, new_y)

            # Check if the move is valid (ignoring whether it's in possible_positions)
            if (
                0 <= new_x < GRID_SIZE and
                0 <= new_y < GRID_SIZE and
                grid[new_x][new_y] == 0 and  # Move to open cells only
                next_pos not in visited  # Avoid revisiting nodes initially
            ):
                visited.add(next_pos)
                new_path = path + [next_pos] if path else [next_pos]
                stack.append((next_pos, new_path))

                # If a move is found, prioritize this path
                if path:
                    chosen_direction = (new_path[0][0] - current_position[0], new_path[0][1] - current_position[1])
                    return new_path[0], True, chosen_direction  # Move to the first step in the new path
                else:
                    return next_pos, True, (dx, dy)  # Move to the new position

    # Step 2: DFS exploration among visited cells (excluding previous_position)
    stack = [(current_position, [])]  # Reinitialize stack for exploring visited cells
    while stack:
        pos, path = stack.pop()

        for (dx, dy), _ in sorted_directions:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            next_pos = (new_x, new_y)

            # Allow movement to visited cells, but exclude previous_position
            if (
                0 <= new_x < GRID_SIZE and
                0 <= new_y < GRID_SIZE and
                grid[new_x][new_y] == 0 and
                next_pos != previous_position  # Exclude the previous position
            ):
                new_path = path + [next_pos] if path else [next_pos]
                stack.append((next_pos, new_path))

                # If a move is found, prioritize this path
                if path:
                    chosen_direction = (new_path[0][0] - current_position[0], new_path[0][1] - current_position[1])
                    return new_path[0], True, chosen_direction  # Move to the first step in the new path
                else:
                    return next_pos, True, (dx, dy)  # Move to the new position

    # Step 3: If no valid move is found, fall back to moving to the previous position
    if previous_position:
        chosen_direction = (previous_position[0] - current_position[0], previous_position[1] - current_position[1])
        return previous_position, True, chosen_direction

    # No valid move found, stay in the current position
    return current_position, False, (0, 0)

def cluster_positions(positions):
    """
    Cluster positions based on their proximity. Uses a simple clustering approach
    by grouping positions that are within a certain distance threshold.
    """
    clusters = []
    threshold = 3  # Distance threshold for clustering

    for pos in positions:
        added = False
        for cluster in clusters:
            # Check if the position is close enough to any existing cluster
            if cdist([pos], cluster, metric='euclidean').min() < threshold:
                cluster.append(pos)
                added = True
                break
        if not added:
            clusters.append([pos])

    return clusters

def move_bot(grid, possible_positions, current_position, previous_position, visited_positions):
    """
    A* algorithm with heuristic for prioritizing movements.
    """
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    def heuristic(position, target_position):
        """Heuristic function: Manhattan distance + open neighbors."""
        x, y = position
        tx, ty = target_position
        distance = abs(x - tx) + abs(y - ty)
        open_neighbors = sum(
            1 for dx, dy in directions
            if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and grid[x + dx][y + dy] == 0
        )
        return distance - open_neighbors  # Prioritize paths with more open neighbors

    if not possible_positions:
        return current_position, False, (0, 0)

    clusters = cluster_positions(possible_positions)
    largest_cluster = max(clusters, key=len)  # Target the largest cluster
    target_position = random.choice(largest_cluster)

    open_list = []
    heappush(open_list, (0, current_position, []))  # (f-cost, position, path)
    g_costs = {current_position: 0}
    visited = set()

    while open_list:
        _, current, path = heappop(open_list)
        if current in visited:
            continue
        visited.add(current)

        if current == target_position:
            # Follow the path to the next position
            if path:
                next_step = path[0]
                visited_positions.add(next_step)
                dx = next_step[0] - current_position[0]
                dy = next_step[1] - current_position[1]
                return next_step, True, (dx, dy)
            else:
                return current_position, False, (0, 0)  # Stay in place if no path

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            new_position = (nx, ny)
            if (
                0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and
                grid[nx, ny] == 0 and new_position not in visited_positions
            ):
                new_g = g_costs[current] + 1  # Each step costs 1
                if new_position not in g_costs or new_g < g_costs[new_position]:
                    g_costs[new_position] = new_g
                    f_cost = new_g + heuristic(new_position, target_position)
                    heappush(open_list, (f_cost, new_position, path + [new_position]))

    # Fallback: No valid move found, stay in place
    return current_position, False, (0, 0)

def plot_grid(ax, grid, current_position, possible_positions, title, eliminated_positions):
    ax.clear()  # Clear the previous plot
    cmap = mcolors.ListedColormap(['white', 'black'])  # Open (0) -> white, Blocked (1) -> black
    ax.imshow(grid, cmap=cmap)

    # Highlight eliminated positions
    for pos in eliminated_positions:
        ax.text(pos[1], pos[0], 'X', color='red', ha='center', va='center', fontsize=10, fontweight='bold')

    # Highlight possible positions
    for pos in possible_positions:
        ax.scatter(pos[1], pos[0], color='yellow', s=100, edgecolor='black', label='Possible Positions' if pos == possible_positions[0] else "")

    # Highlight current position
    ax.scatter(current_position[1], current_position[0], color='red', s=100, edgecolor='black', label='Bot Current Position')

    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.invert_yaxis()  # Invert y-axis for intuitive visualization
    plt.draw()
    plt.pause(0.1)  # Pause to visualize the update

def update_space_rat_knowledge_base(grid, bot_position, rat_positions, alpha, ping_received, state):
    """
    Update possible positions of the space rat using Bayesian inference based on sensor data.
    """
    # global SPACE_RAT_DETECTOR_ACTIONS
    # SPACE_RAT_DETECTOR_ACTIONS = SPACE_RAT_DETECTOR_ACTIONS + 1
    state.space_rat_detector_actions += 1
    state.actions.append("detection")
    bx, by = bot_position
    updated_rat_positions = []
    posterior_probabilities = {}

    # Compute prior probabilities
    total_prior = len(rat_positions)
    prior_prob = 1 / total_prior if total_prior > 0 else 0

    for rx, ry in rat_positions:
        distance = abs(bx - rx) + abs(by - ry)  # Manhattan distance
        likelihood = math.exp(-alpha * (distance - 1)) if ping_received else 1 - math.exp(-alpha * (distance - 1))
        
        # Bayes' theorem: posterior = (likelihood * prior) / normalization constant
        posterior_prob = likelihood * prior_prob
        posterior_probabilities[(rx, ry)] = posterior_prob

    # Normalize probabilities
    total_posterior = sum(posterior_probabilities.values()) or 1
    normalized_probabilities = {pos: prob / total_posterior for pos, prob in posterior_probabilities.items()}

    # Update the list of possible rat positions based on the new probabilities
    updated_rat_positions = [pos for pos, prob in normalized_probabilities.items() if prob > 0]

    return updated_rat_positions, normalized_probabilities

def manhattan_distance(pos1, pos2):
    """Helper function to compute Manhattan distance."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Preferred for now for static rat
def markov_filter_update(grid, possible_rat_positions, current_position, alpha, ping_received, state, rat_probabilities):
    new_rat_probabilities = {}
    state.space_rat_detector_actions += 1
    state.actions.append("detection")

    # Update probabilities based on the received ping and transition dynamics (if applicable)
    for x, y in possible_rat_positions:
        distance = abs(x - current_position[0]) + abs(y - current_position[1])
        probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1
        if ping_received:
            new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * probability_of_ping
        else:
            new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * (1 - probability_of_ping)

    # Normalize probabilities
    total_prob = sum(new_rat_probabilities.values())
    for key in new_rat_probabilities:
        new_rat_probabilities[key] = (new_rat_probabilities[key] / total_prob) if total_prob > 0 else 1 / len(possible_rat_positions)

    return new_rat_probabilities

def markov_filter_update_for_utility_calc(grid, possible_rat_positions, current_position, alpha, ping_received, rat_probabilities):
    new_rat_probabilities = {}

    # Update probabilities based on the received ping and transition dynamics (if applicable)
    for x, y in possible_rat_positions:
        distance = abs(x - current_position[0]) + abs(y - current_position[1])
        probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1
        if ping_received:
            new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * probability_of_ping
        else:
            new_rat_probabilities[(x, y)] = rat_probabilities.get((x, y), 1 / len(possible_rat_positions)) * (1 - probability_of_ping)

    # Normalize probabilities
    total_prob = sum(new_rat_probabilities.values())
    for key in new_rat_probabilities:
        new_rat_probabilities[key] = (new_rat_probabilities[key] / total_prob) if total_prob > 0 else 1 / len(possible_rat_positions)

    return new_rat_probabilities

# For dynamic Rat
def markov_filter_update_with_prediction(grid, possible_rat_positions, current_position, alpha, ping_received, state, rat_probabilities):
    new_rat_probabilities = {}
    state.space_rat_detector_actions += 1
    state.actions.append("detection")

    # Predict rat movement based on a simple Markov model (random movement assumed for this example)
    transition_probabilities = {}  # Dictionary to store transition probabilities for each position
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Cardinal directions (right, left, down, up)

    for x, y in possible_rat_positions:
        valid_moves = [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and grid[x + dx][y + dy] == 0]
        num_valid_moves = len(valid_moves)
        if num_valid_moves > 0:
            transition_probabilities[(x, y)] = {move: 1 / num_valid_moves for move in valid_moves}

    # Update probabilities based on ping received and predicted rat movement
    for (x, y), prob in rat_probabilities.items():
        new_prob = prob
        distance = abs(x - current_position[0]) + abs(y - current_position[1])
        probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1

        # Apply ping update
        if ping_received:
            new_prob *= probability_of_ping
        else:
            new_prob *= (1 - probability_of_ping)

        # Predictive update based on transition probabilities
        if (x, y) in transition_probabilities:
            for next_position, transition_prob in transition_probabilities[(x, y)].items():
                if next_position in new_rat_probabilities:
                    new_rat_probabilities[next_position] += new_prob * transition_prob
                else:
                    new_rat_probabilities[next_position] = new_prob * transition_prob

    # Normalize probabilities
    total_prob = sum(new_rat_probabilities.values())
    for key in new_rat_probabilities:
        new_rat_probabilities[key] = (new_rat_probabilities[key] / total_prob) if total_prob > 0 else 1 / len(possible_rat_positions)

    return new_rat_probabilities

def markov_filter_update_with_prediction_for_utility(grid, possible_rat_positions, current_position, alpha, ping_received, rat_probabilities):
    new_rat_probabilities = {}

    # Predict rat movement based on a simple Markov model (random movement assumed for this example)
    transition_probabilities = {}  # Dictionary to store transition probabilities for each position
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Cardinal directions (right, left, down, up)

    for x, y in possible_rat_positions:
        valid_moves = [(x + dx, y + dy) for dx, dy in directions if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and grid[x + dx][y + dy] == 0]
        num_valid_moves = len(valid_moves)
        if num_valid_moves > 0:
            transition_probabilities[(x, y)] = {move: 1 / num_valid_moves for move in valid_moves}

    # Update probabilities based on ping received and predicted rat movement
    for (x, y), prob in rat_probabilities.items():
        new_prob = prob
        distance = abs(x - current_position[0]) + abs(y - current_position[1])
        probability_of_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1

        # Apply ping update
        if ping_received:
            new_prob *= probability_of_ping
        else:
            new_prob *= (1 - probability_of_ping)

        # Predictive update based on transition probabilities
        if (x, y) in transition_probabilities:
            for next_position, transition_prob in transition_probabilities[(x, y)].items():
                if next_position in new_rat_probabilities:
                    new_rat_probabilities[next_position] += new_prob * transition_prob
                else:
                    new_rat_probabilities[next_position] = new_prob * transition_prob

    # Normalize probabilities
    total_prob = sum(new_rat_probabilities.values())
    for key in new_rat_probabilities:
        new_rat_probabilities[key] = (new_rat_probabilities[key] / total_prob) if total_prob > 0 else 1 / len(possible_rat_positions)

    return new_rat_probabilities

def heuristic_rat(position, target_position, position_weights):
    """
    Heuristic function for A* algorithm that considers both Manhattan distance
    and position weights (probability of finding the rat).
    """
    px, py = position
    tx, ty = target_position
    distance = abs(px - tx) + abs(py - ty)  # Manhattan distance
    weight = position_weights.get(position, 0)
    # The heuristic is a combination of distance and a preference for higher weights
    return distance - weight  # Higher weight positions are prioritized

def move_bot_towards_rat(grid, bot_position, possible_rat_positions, position_weights, visited_positions):
    if not possible_rat_positions:
        return bot_position, False

    bx, by = bot_position
    target_positions = sorted(possible_rat_positions, key=lambda pos: position_weights.get(pos, 0), reverse=True)
    target_position = target_positions[0]  # Choose the highest probability target as the main goal

    # A* Search Initialization
    open_set = [(0, bot_position, [])]  # (f-score, position, path)
    g_scores = {bot_position: 0}  # Cost from start to the current position
    f_scores = {bot_position: heuristic_rat(bot_position, target_position, position_weights)}  # f = g + h
    visited = set()

    while open_set:
        # Get the position with the lowest f-score
        _, current_position, path = min(open_set, key=lambda x: x[0])
        open_set = [entry for entry in open_set if entry[1] != current_position]  # Remove it from the open set

        if current_position in possible_rat_positions and current_position not in visited_positions:
            # If this position is one of the targets and not visited, return the path
            if path:
                next_step = path[0]
                return next_step, True
            else:
                return current_position, True  # Directly at target position

        visited.add(current_position)
        cx, cy = current_position

        # Generate neighbors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            new_x, new_y = cx + dx, cy + dy
            new_position = (new_x, new_y)

            if (
                0 <= new_x < GRID_SIZE and
                0 <= new_y < GRID_SIZE and
                grid[new_x][new_y] == 0 and  # Check if the cell is open
                new_position not in visited
            ):
                tentative_g_score = g_scores[current_position] + 1  # Assume uniform cost for all moves
                if tentative_g_score < g_scores.get(new_position, float('inf')):
                    # This path to new_position is better than any previous one
                    g_scores[new_position] = tentative_g_score
                    f_score = tentative_g_score + heuristic_rat(new_position, target_position, position_weights)
                    f_scores[new_position] = f_score
                    open_set.append((f_score, new_position, path + [new_position]))

    # Fallback: Make a random valid move if stuck
    for dx, dy in random.sample([(0, 1), (0, -1), (1, 0), (-1, 0)], 4):
        new_x, new_y = bx + dx, by + dy
        if (new_x, new_y) not in visited_positions and 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and grid[new_x][new_y] == 0:
            print("Fallback move to avoid getting stuck")
            return (new_x, new_y), True

    return bot_position, False  # No valid movement possible

def move_rat_randomly(grid, rat_position):
    x, y = rat_position
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up
    open_directions = [
        (x + dx, y + dy) for dx, dy in directions
        if 0 <= x + dx < GRID_SIZE and 0 <= y + dy < GRID_SIZE and grid[x + dx][y + dy] == 0
    ]
    if open_directions:
        return random.choice(open_directions)
    return rat_position  # If no open directions, stay in the same place

def plot_movement_vs_detection(actions, alpha, x=25):
    """
    Plots the last x actions (movement or detection) performed sequentially over time until the rat is caught.
    Each action is represented by a bar in the bar graph: movement (blue) and detection (red).
    The plot is saved to an image file named 'last_25_actions_<alpha_value>.png'.
    """
    # Take only the last x actions
    actions_to_plot = actions[-x:] if len(actions) > x else actions
    timestamps = range(1, len(actions_to_plot) + 1)  # Generate sequential timestamps from 1 to len(actions_to_plot)

    # Assign colors based on the action type
    colors = ['blue' if action == 'movement' else 'red' for action in actions_to_plot]

    plt.figure(figsize=(10, 6))
    plt.bar(timestamps, [1] * len(actions_to_plot), color=colors, alpha=0.7, edgecolor='black')  # Uniform bar heights of 1
    plt.xlabel('Timestamp')
    plt.ylabel('Action Type (Height Uniform)')
    plt.title(f'Movement vs Rat Detection Over Last {len(actions_to_plot)} Actions for Alpha {alpha}')
    plt.xticks(timestamps)  # Display all timestamps on the x-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add a grid for y-axis
    plt.tight_layout()  # Adjust the layout for better fit

    # Save the plot to a file
    filename = f'last_25_actions_{alpha:.2f}.png'
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory

def scale_pulse_weight(pulse_rate, alpha):
    # Reduce pulse weight influence when alpha is close to 0 or 1
    if alpha < 0.1 or alpha > 0.9:
        return pulse_rate * 0.5  # Scale down influence (adjust factor as needed)
    return pulse_rate

# Prepare data from aggregated logs
# def prepare_data_from_logs(all_data_logs, grid_size):
#     inputs = []
#     targets = []
#     timestamps = []

#     for log in all_data_logs:
#         inputs.append(log["input_tensor"].numpy())  # Convert tensor to numpy array
#         targets.append(log["total_actions"])
#         timestamps.append(log["timestamp"])

#     inputs = torch.tensor(inputs, dtype=torch.float32)  # Convert back to tensor
#     targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  # Targets must be 2D for MSELoss
#     timestamps = torch.tensor(timestamps, dtype=torch.float32)

#     return inputs, targets, timestamps

# Prepare data from aggregated logs
def prepare_data_from_logs(all_data_logs, grid_size):
    inputs = []
    targets = []
    timestamps = []

    for log in all_data_logs:
        # Extract the updated input_tensor
        input_tensor = log["input_tensor"].numpy()  # Convert tensor to numpy array
        if input_tensor.shape[0] != 3:  # Ensure the input has 3 channels
            print(f"Unexpected input shape: {input_tensor.shape}. Expected 3 channels (rat_probs, ship_layout, bot_position).")
            # raise ValueError(f"Unexpected input shape: {input_tensor.shape}. Expected 3 channels (rat_probs, ship_layout, bot_position).")
        inputs.append(input_tensor)
        targets.append(log["total_actions"])
        timestamps.append(log["timestamp"])

    # Convert lists to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)  # Shape: [batch_size, 3, grid_size, grid_size]
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  # Shape: [batch_size, 1]
    timestamps = torch.tensor(timestamps, dtype=torch.float32)  # Shape: [batch_size]

    return inputs, targets, timestamps


# Split data into training and testing sets
def split_data(inputs, targets, timestamps, test_size=0.2):
    train_inputs, test_inputs, train_targets, test_targets, train_timestamps, test_timestamps = train_test_split(
        inputs, targets, timestamps, test_size=test_size, random_state=42
    )
    return train_inputs, test_inputs, train_targets, test_targets, train_timestamps, test_timestamps

# def train_model(model, train_inputs, train_targets, test_inputs, test_targets, epochs, batch_size, learning_rate):
#     loss_function = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     train_losses = []
#     test_losses = []

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0

#         # Shuffle data for each epoch
#         perm = torch.randperm(train_inputs.size(0))
#         train_inputs = train_inputs[perm]
#         train_targets = train_targets[perm]

#         for i in range(0, train_inputs.size(0), batch_size):
#             batch_inputs = train_inputs[i:i+batch_size]
#             batch_targets = train_targets[i:i+batch_size]

#             optimizer.zero_grad()
#             predictions = model(batch_inputs)
#             loss = loss_function(predictions, batch_targets)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_train_loss = total_loss / (train_inputs.size(0) // batch_size)
#         train_losses.append(avg_train_loss)

#         # Evaluate on testing data
#         model.eval()
#         with torch.no_grad():
#             test_predictions = model(test_inputs)
#             test_loss = loss_function(test_predictions, test_targets).item()
#             test_losses.append(test_loss)

#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}")

#     return train_losses, test_losses

# def prepare_input_tensor(grid, bot_position):
#     """
#     Prepares the input tensor for the model based on the current grid state and bot position.

#     :param grid: 2D numpy array representing the grid layout.
#     :param bot_position: Tuple (x, y) representing the bot's position.
#     :return: PyTorch tensor with shape [channels, grid_size, grid_size].
#     """
#     GRID_SIZE = grid.shape[0]  # Assuming a square grid

#     # Channel 1: Rat probabilities (initialize with zeros or pass probabilities if available)
#     rat_probabilities = np.zeros((GRID_SIZE, GRID_SIZE))

#     # Channel 2: Grid layout (walls = 1, open spaces = 0)
#     ship_layout = (grid > 0).astype(float)  # Convert to binary (1 for walls, 0 for open)

#     # Channel 3: Bot position (binary grid with 1 at the bot's position)
#     bot_position_grid = np.zeros((GRID_SIZE, GRID_SIZE))
#     bot_position_grid[bot_position[0], bot_position[1]] = 1  # Mark bot's position as 1

#     # Stack channels into a single tensor
#     input_tensor = np.stack([rat_probabilities, ship_layout, bot_position_grid])
#     input_tensor = torch.tensor(input_tensor, dtype=torch.float32)  # Convert to PyTorch tensor

#     return input_tensor

def prepare_input_tensor(grid, bot_position, rat_probabilities):
    """
    Prepares the input tensor for the model based on the current grid state, bot position, and rat probabilities.

    :param grid: 2D numpy array representing the grid layout.
    :param bot_position: Tuple (x, y) representing the bot's position.
    :param rat_probabilities: Dictionary mapping positions to probabilities.
    :return: PyTorch tensor with shape [channels, grid_size, grid_size].
    """
    GRID_SIZE = grid.shape[0]  # Assuming a square grid

    # Channel 1: Rat probabilities (populate based on the provided dictionary)
    rat_probabilities_array = np.zeros((GRID_SIZE, GRID_SIZE))
    for (x, y), prob in rat_probabilities.items():
        rat_probabilities_array[x, y] = prob

    # Channel 2: Grid layout (walls = 1, open spaces = 0)
    ship_layout = (grid > 0).astype(float)  # Convert to binary (1 for walls, 0 for open)

    # Channel 3: Bot position (binary grid with 1 at the bot's position)
    bot_position_grid = np.zeros((GRID_SIZE, GRID_SIZE))
    bot_position_grid[bot_position[0], bot_position[1]] = 1  # Mark bot's position as 1

    # Stack channels into a single tensor
    input_tensor = np.stack([rat_probabilities_array, ship_layout, bot_position_grid])
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32)  # Convert to PyTorch tensor

    return input_tensor


def simulate_action(grid, current_position, action):
    x, y = current_position
    if action == "up" and x > 0 and grid[x-1][y] == 0:
        return (x-1, y)
    elif action == "down" and x < GRID_SIZE-1 and grid[x+1][y] == 0:
        return (x+1, y)
    elif action == "left" and y > 0 and grid[x][y-1] == 0:
        return (x, y-1)
    elif action == "right" and y < GRID_SIZE-1 and grid[x][y+1] == 0:
        return (x, y+1)
    return None  # Invalid action

# def evaluate_actions(grid, model, bot_position, possible_actions):
#     action_scores = {}
#     for action in possible_actions:
#         next_position = simulate_action(grid, bot_position, action)
#         if next_position:
#             # Prepare the input tensor for the model
#             input_tensor = prepare_input_tensor(grid, next_position)  # Add your specific function here
            
#             # Predict using the model
#             with torch.no_grad():
#                 score = model(input_tensor).item()
#             action_scores[action] = score
#     return action_scores

def evaluate_actions(alpha, distance, grid, model, bot_position, possible_actions, rat_probabilities, rat_probabilities_latest, possible_rat_positions):
    """
    Evaluates possible actions and returns a score for each action based on the model's predictions.

    :param grid: 2D numpy array representing the grid layout.
    :param model: Trained model for evaluating actions.
    :param bot_position: Tuple (x, y) representing the bot's position.
    :param possible_actions: List of possible actions (e.g., "up", "down", "left", "right", "sense").
    :return: Dictionary with actions as keys and scores as values.
    """
    action_scores = {}
    for action in possible_actions:
        if action == "sense":
            # For "sense", the bot stays in place
            input_tensor = prepare_input_tensor(grid, bot_position, rat_probabilities_latest)
        else:
            # For movement actions, simulate the next position
            next_position = simulate_action(grid, bot_position, action)
            if not next_position:
                continue  # Skip invalid moves

            # Simulate knowledge base update using a copy
            temp_rat_probabilities = rat_probabilities.copy()

            # Calculate the probability of receiving a ping
            prob_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1
            updated_probs_ping = markov_filter_update_for_utility_calc(grid, possible_rat_positions, next_position, alpha, True, temp_rat_probabilities)
            updated_probs_no_ping = markov_filter_update_for_utility_calc(grid, possible_rat_positions, next_position, alpha, False, temp_rat_probabilities)

            # Weighted average of probabilities based on the likelihood of receiving a ping
            simulated_rat_probabilities = {
                pos: prob_ping * updated_probs_ping.get(pos, 0) + (1 - prob_ping) * updated_probs_no_ping.get(pos, 0)
                for pos in possible_rat_positions
            }

            # Normalize the probabilities
            total_prob = sum(simulated_rat_probabilities.values())
            if total_prob > 0:
                simulated_rat_probabilities = {pos: p / total_prob for pos, p in simulated_rat_probabilities.items()}
            
            input_tensor = prepare_input_tensor(grid, next_position, simulated_rat_probabilities)
        
        # Predict using the model
        with torch.no_grad():
            score = model(input_tensor.unsqueeze(0)).item()  # Add batch dimension for the model
        action_scores[action] = score

    return action_scores


def compare_model_with_bot(alpha, distance, grid, model, bot_position, actual_action, possible_actions, rat_probabilities, rat_probabilities_latest, possible_rat_positions):
    action_scores = evaluate_actions(alpha, distance, grid, model, bot_position, possible_actions, rat_probabilities, rat_probabilities_latest, possible_rat_positions)
    best_action = min(action_scores, key=action_scores.get)  # Assuming lower score is better

    # Compare the best action with the bot's actual action
    return best_action == actual_action, best_action, action_scores


def train_model(model, train_inputs, train_targets, test_inputs, test_targets, epochs, batch_size, learning_rate):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    scaled_train_losses = []
    scaled_test_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Shuffle data for each epoch
        perm = torch.randperm(train_inputs.size(0))
        train_inputs = train_inputs[perm]
        train_targets = train_targets[perm]

        for i in range(0, train_inputs.size(0), batch_size):
            batch_inputs = train_inputs[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]

            optimizer.zero_grad()
            predictions = model(batch_inputs)

            # Calculate loss normally
            loss = loss_function(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / (train_inputs.size(0) // batch_size)
        train_losses.append(avg_train_loss)

        # Evaluate on testing data
        model.eval()
        with torch.no_grad():
            test_predictions = model(test_inputs)
            test_loss = loss_function(test_predictions, test_targets).item()

        test_losses.append(test_loss)

        # Print scaled loss for logging, if needed
        scaled_train_loss = avg_train_loss if avg_train_loss <= 500 else 10
        scaled_train_losses.append(scaled_train_loss)
        scaled_test_loss = test_loss if test_loss <= 500 else 10
        scaled_test_losses.append(scaled_test_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss (Scaled for Logging): {scaled_train_loss:.4f} | Test Loss (Scaled for Logging): {scaled_test_loss:.4f}")

    return train_losses, test_losses, scaled_train_losses, scaled_test_losses

# def train_model(model, train_inputs, train_targets, test_inputs, test_targets, epochs, batch_size, learning_rate):
#     loss_function = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     train_losses = []
#     test_losses = []
#     scaled_train_losses = []
#     scaled_test_losses = []

#     # Verify dataset sizes
#     assert train_inputs.size(0) == train_targets.size(0), \
#         f"Mismatch in sizes: train_inputs={train_inputs.size(0)}, train_targets={train_targets.size(0)}"
#     assert test_inputs.size(0) == test_targets.size(0), \
#         f"Mismatch in sizes: test_inputs={test_inputs.size(0)}, test_targets={test_targets.size(0)}"

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0

#         # Shuffle data for each epoch
#         perm_size = min(train_inputs.size(0), train_targets.size(0))  # Ensure no out-of-bounds indexing
#         perm = torch.randperm(perm_size)
#         train_inputs = train_inputs[perm]
#         train_targets = train_targets[perm]

#         for i in range(0, train_inputs.size(0), batch_size):
#             batch_inputs = train_inputs[i:i+batch_size]
#             batch_targets = train_targets[i:i+batch_size]

#             optimizer.zero_grad()
#             predictions = model(batch_inputs)

#             # Calculate loss
#             loss = loss_function(predictions, batch_targets)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_train_loss = total_loss / (train_inputs.size(0) // batch_size)
#         train_losses.append(avg_train_loss)

#         # Evaluate on testing data
#         model.eval()
#         with torch.no_grad():
#             test_predictions = model(test_inputs)
#             test_loss = loss_function(test_predictions, test_targets).item()
#         test_losses.append(test_loss)

#         # Scale losses for logging
#         scaling_threshold = 100  # Adjust scaling if needed
#         scaled_train_loss = avg_train_loss if avg_train_loss <= scaling_threshold else scaling_threshold
#         scaled_test_loss = test_loss if test_loss <= scaling_threshold else scaling_threshold
#         scaled_train_losses.append(scaled_train_loss)
#         scaled_test_losses.append(scaled_test_loss)

#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}")

#     return train_losses, test_losses, scaled_train_losses, scaled_test_losses


# def train_model(model, train_inputs, train_targets, test_inputs, test_targets, epochs, batch_size, learning_rate):
#     loss_function = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     train_losses = []  # Original training losses
#     test_losses = []   # Original testing losses
#     scaled_train_losses = []  # Scaled training losses
#     scaled_test_losses = []   # Scaled testing losses

#     # Define scaling parameters
#     min_target = 10
#     max_target = 20
#     original_min = 50
#     original_max = 1000  # Adjust based on expected max loss

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0

#         # Shuffle data for each epoch
#         perm = torch.randperm(train_inputs.size(0))
#         train_inputs = train_inputs[perm]
#         train_targets = train_targets[perm]

#         for i in range(0, train_inputs.size(0), batch_size):
#             batch_inputs = train_inputs[i:i+batch_size]
#             batch_targets = train_targets[i:i+batch_size]

#             optimizer.zero_grad()
#             predictions = model(batch_inputs)
#             loss = loss_function(predictions, batch_targets)
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_train_loss = total_loss / (train_inputs.size(0) // batch_size)
#         train_losses.append(avg_train_loss)  # Log the original training loss

#         # Scale the training loss if it exceeds 100
#         if avg_train_loss > original_min:
#             scaled_train_loss = min_target + (
#                 (avg_train_loss - original_min) / (original_max - original_min)
#             ) * (max_target - min_target)
#             scaled_train_loss = min(max_target, max(min_target, scaled_train_loss))  # Ensure within bounds
#         else:
#             scaled_train_loss = avg_train_loss

#         scaled_train_losses.append(scaled_train_loss)  # Log the scaled training loss

#         # Evaluate on testing data
#         model.eval()
#         with torch.no_grad():
#             test_predictions = model(test_inputs)
#             test_loss = loss_function(test_predictions, test_targets).item()

#             test_losses.append(test_loss)  # Log the original testing loss

#             # Scale the test loss similarly
#             if test_loss > original_min:
#                 scaled_test_loss = min_target + (
#                     (test_loss - original_min) / (original_max - original_min)
#                 ) * (max_target - min_target)
#                 scaled_test_loss = min(max_target, max(min_target, scaled_test_loss))  # Ensure within bounds
#             else:
#                 scaled_test_loss = test_loss

#             scaled_test_losses.append(scaled_test_loss)  # Log the scaled testing loss

#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} (Scaled: {scaled_train_loss:.4f}) | "
#               f"Test Loss: {test_loss:.4f} (Scaled: {scaled_test_loss:.4f})")

#     return train_losses, test_losses, scaled_train_losses, scaled_test_losses


# Dynamic Latest
# def main_phase_2(grid, initial_bot_position, rat_initial_position, alpha, state, model, loss_function, loss_log):
#     timestamp = 0
#     timeout_limit = 9000
#     possible_rat_positions = [(x, y) for x in range(1, GRID_SIZE - 1) for y in range(1, GRID_SIZE - 1) if grid[x][y] == 0]
#     bot_position = initial_bot_position
#     rat_position = rat_initial_position
#     visited_positions = set()
#     rat_probabilities = {pos: 1 / len(possible_rat_positions) for pos in possible_rat_positions}

#     # UCB-related variables
#     action_counts = {'movement': 1, 'sensing': 1}  # Initialize with 1 to avoid division by zero
#     action_rewards = {'movement': 0.0, 'sensing': 0.0}
#     c = 2  # Exploration parameter
#     print(f"c value: {c}")

#     while state.movements < timeout_limit:
#         timestamp += 1
#         # Move the rat randomly in one of the open directions
#         rat_position = move_rat_randomly(grid, rat_position)
#         bx, by = bot_position
#         rx, ry = rat_position
#         distance = abs(bx - rx) + abs(by - ry)
#         pulse_rate = max(1.0 / (distance + 1), 0.1)
#         timestamp = 0

#         # Determine if a ping is received based on the detection probability
#         if distance == 0:
#             ping_received = True
#         else:
#             probability_of_ping = math.exp(-alpha * (distance - 1))
#             ping_received = random.random() < probability_of_ping

#         # Calculate UCB values for each action
#         total_actions = sum(action_counts.values())
#         ucb_movement = (action_rewards['movement'] / action_counts['movement']) + \
#                        c * math.sqrt(math.log(total_actions) / action_counts['movement'])
#         ucb_sensing = (action_rewards['sensing'] / action_counts['sensing']) + \
#                       c * math.sqrt(math.log(total_actions) / action_counts['sensing'])

#         # Choose the action with the highest UCB value
#         if ucb_movement > ucb_sensing:
#             # Perform movement
#             most_probable_rat_position = max(rat_probabilities, key=rat_probabilities.get)
#             # most_probable_rat_position = rat_position # Using actual rat position
#             possible_rat_positions = [pos for pos in possible_rat_positions if rat_probabilities.get(pos, 0) > 0]

#             if most_probable_rat_position in possible_rat_positions:
#                 possible_rat_positions = [most_probable_rat_position] + [pos for pos in possible_rat_positions if pos != most_probable_rat_position]

#             next_position, moved = move_bot_towards_rat(grid, bot_position, possible_rat_positions, rat_probabilities, visited_positions)

#             if moved:
#                 state.movements += 1
#                 state.actions.append("movement")
#                 bot_position = next_position
#                 visited_positions.add(bot_position)

#                 # Calculate total probability before movement
#                 total_probability_before = sum(rat_probabilities.get(pos, 0) for pos in possible_rat_positions)

#                 # Simulate knowledge base update using a copy
#                 temp_rat_probabilities = rat_probabilities.copy()

#                 # Calculate the probability of receiving a ping
#                 prob_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1
#                 updated_probs_ping = markov_filter_update_with_prediction_for_utility(grid, possible_rat_positions, bot_position, alpha, True, temp_rat_probabilities)
#                 updated_probs_no_ping = markov_filter_update_with_prediction_for_utility(grid, possible_rat_positions, bot_position, alpha, False, temp_rat_probabilities)

#                 # Weighted average of probabilities based on the likelihood of receiving a ping
#                 simulated_rat_probabilities = {
#                     pos: prob_ping * updated_probs_ping.get(pos, 0) + (1 - prob_ping) * updated_probs_no_ping.get(pos, 0)
#                     for pos in possible_rat_positions
#                 }

#                 # Normalize the probabilities
#                 total_prob = sum(simulated_rat_probabilities.values())
#                 if total_prob > 0:
#                     simulated_rat_probabilities = {pos: p / total_prob for pos, p in simulated_rat_probabilities.items()}

#                 # Calculate total probability after movement using simulated probabilities
#                 total_probability_after = sum(simulated_rat_probabilities.get(pos, 0) for pos in possible_rat_positions)

#                 # Reward based on probability improvement
#                 max_possible_prob_change = 1
#                 reward = max(0, total_probability_after - total_probability_before) / max_possible_prob_change

#                 # reward = max(0, total_probability_after - total_probability_before)
#                 action_rewards['movement'] += reward
#                 action_counts['movement'] += 1
#         else:
#             # Perform sensing
#             old_entropy = -sum(p * math.log(p + 1e-9) for p in rat_probabilities.values())  # Calculate entropy
#             rat_probabilities = markov_filter_update_with_prediction(grid, possible_rat_positions, bot_position, alpha, ping_received, state, rat_probabilities)
#             new_entropy = -sum(p * math.log(p + 1e-9) for p in rat_probabilities.values())  # Calculate new entropy
#             max_entropy = math.log(len(possible_rat_positions))  # Maximum possible entropy
#             reward = max(0, (old_entropy - new_entropy) / max_entropy)  # Positive reward if entropy decreases
#             # reward = max(0, old_entropy - new_entropy)  # Positive reward if entropy decreases
#             action_rewards['sensing'] += reward
#             action_counts['sensing'] += 1

#         # Check if bot has caught the space rat
#         if bot_position == rat_position:
#             print(f"Bot caught the space rat at {bot_position}", flush=True)
#             break

#         # Increment timestamp for each action
#         timestamp += 1

#         # If model is provided, predict and calculate loss
#         if model and loss_function is not None:
#             # Flatten grid and include timestamp as input
#             flattened_grid = grid.flatten()
#             input_vector = torch.tensor(np.concatenate((flattened_grid, [timestamp])), dtype=torch.float32).unsqueeze(0)

#             # Model prediction
#             prediction = model(input_vector)
#             actual_output = state.movements + state.space_rat_detector_actions

#             # Calculate loss and log it
#             loss = loss_function(prediction, torch.tensor([[actual_output]], dtype=torch.float32))
#             loss_log.append((timestamp, loss.item()))

#         # Stop condition if rat is caught
#         if bot_position == rat_position:
#             print(f"Bot caught the space rat at {bot_position}")
#             break

#     if state.movements >= timeout_limit:
#         state.movements = 0
#         state.blocked_cell_actions = 0
#         state.space_rat_detector_actions = 0
#         print("Failed Simulation for catching rat")
#         state.failed_simulations += 1

# Static Latest
def main_phase_2(grid, initial_bot_position, rat_initial_position, alpha, state, data_log, model):
    timestamp = 0
    timeout_limit = 9000
    possible_rat_positions = [(x, y) for x in range(1, GRID_SIZE - 1) for y in range(1, GRID_SIZE - 1) if grid[x][y] == 0]
    bot_position = initial_bot_position
    rat_position = rat_initial_position
    visited_positions = set()
    rat_probabilities = {pos: 1 / len(possible_rat_positions) for pos in possible_rat_positions}

    # UCB-related variables
    action_counts = {'movement': 1, 'sensing': 1}  # Initialize with 1 to avoid division by zero
    action_rewards = {'movement': 0.0, 'sensing': 0.0}
    c = 2  # Exploration parameter
    print(f"c value: {c}")

    # Possible actions
    possible_actions = ["up", "down", "left", "right", "sense"]
    action_score_differences = []
    all_action_scores = []
    agreement_log = []
    metrics_log = []
    simulation_metrics = {}

    while state.movements < timeout_limit:
        timestamp += 1
        actual_action = None
        # Move the rat randomly in one of the open directions
        # rat_position = move_rat_randomly(grid, rat_position)
        bx, by = bot_position
        rx, ry = rat_position
        distance = abs(bx - rx) + abs(by - ry)
        pulse_rate = max(1.0 / (distance + 1), 0.1)

        # Determine if a ping is received based on the detection probability
        if distance == 0:
            ping_received = True
        else:
            probability_of_ping = math.exp(-alpha * (distance - 1))
            ping_received = random.random() < probability_of_ping

        # Calculate UCB values for each action
        total_actions = sum(action_counts.values())
        ucb_movement = (action_rewards['movement'] / action_counts['movement']) + \
                       c * math.sqrt(math.log(total_actions) / action_counts['movement'])
        ucb_sensing = (action_rewards['sensing'] / action_counts['sensing']) + \
                      c * math.sqrt(math.log(total_actions) / action_counts['sensing'])

        # storing prev position for model evaluation
        prev_bot_position = bot_position
        # Create a copy of the rat probabilities array
        rat_probabilities_copy = rat_probabilities.copy()

        # Choose the action with the highest UCB value
        if ucb_movement > ucb_sensing:
            # Perform movement
            most_probable_rat_position = max(rat_probabilities, key=rat_probabilities.get)
            # most_probable_rat_position = rat_position # Using actual rat position
            possible_rat_positions = [pos for pos in possible_rat_positions if rat_probabilities.get(pos, 0) > 0]

            if most_probable_rat_position in possible_rat_positions:
                possible_rat_positions = [most_probable_rat_position] + [pos for pos in possible_rat_positions if pos != most_probable_rat_position]

            next_position, moved = move_bot_towards_rat(grid, bot_position, possible_rat_positions, rat_probabilities, visited_positions)

            if moved:
                state.movements += 1
                state.actions.append("movement")

                # determine direction for model vs actual action
                bx, by = bot_position
                nx, ny = next_position

                if nx == bx - 1 and ny == by:
                    actual_action = 'up'
                elif nx == bx + 1 and ny == by:
                    actual_action = 'down'
                elif nx == bx and ny == by - 1:
                    actual_action = 'left'
                elif nx == bx and ny == by + 1:
                    actual_action = 'right'

                bot_position = next_position
                visited_positions.add(bot_position)

                # Calculate total probability before movement
                total_probability_before = sum(rat_probabilities.get(pos, 0) for pos in possible_rat_positions)

                # Simulate knowledge base update using a copy
                temp_rat_probabilities = rat_probabilities.copy()

                # Calculate the probability of receiving a ping
                prob_ping = math.exp(-alpha * (distance - 1)) if distance > 0 else 1
                updated_probs_ping = markov_filter_update_for_utility_calc(grid, possible_rat_positions, bot_position, alpha, True, temp_rat_probabilities)
                updated_probs_no_ping = markov_filter_update_for_utility_calc(grid, possible_rat_positions, bot_position, alpha, False, temp_rat_probabilities)

                # Weighted average of probabilities based on the likelihood of receiving a ping
                simulated_rat_probabilities = {
                    pos: prob_ping * updated_probs_ping.get(pos, 0) + (1 - prob_ping) * updated_probs_no_ping.get(pos, 0)
                    for pos in possible_rat_positions
                }

                # Normalize the probabilities
                total_prob = sum(simulated_rat_probabilities.values())
                if total_prob > 0:
                    simulated_rat_probabilities = {pos: p / total_prob for pos, p in simulated_rat_probabilities.items()}

                # Calculate total probability after movement using simulated probabilities
                total_probability_after = sum(simulated_rat_probabilities.get(pos, 0) for pos in possible_rat_positions)

                # Reward based on probability improvement
                max_possible_prob_change = 1
                reward = max(0, total_probability_after - total_probability_before) / max_possible_prob_change

                # reward = max(0, total_probability_after - total_probability_before)
                action_rewards['movement'] += reward
                action_counts['movement'] += 1
        else:
            # Perform sensing
            actual_action = 'sense'
            old_entropy = -sum(p * math.log(p + 1e-9) for p in rat_probabilities.values())  # Calculate entropy
            rat_probabilities = markov_filter_update(grid, possible_rat_positions, bot_position, alpha, ping_received, state, rat_probabilities)
            new_entropy = -sum(p * math.log(p + 1e-9) for p in rat_probabilities.values())  # Calculate new entropy
            max_entropy = math.log(len(possible_rat_positions))  # Maximum possible entropy
            reward = max(0, (old_entropy - new_entropy) / max_entropy)  # Positive reward if entropy decreases
            # reward = max(0, old_entropy - new_entropy)  # Positive reward if entropy decreases
            action_rewards['sensing'] += reward
            action_counts['sensing'] += 1

        # Check if a saved model exists
        model_path = "space_rat_model.pth"
        if os.path.exists(model_path):
            print("Loading saved model...")
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print("Evaluating model with actual action ...")
            # Compare model prediction with bot's action
            model_agrees, best_action, action_scores = compare_model_with_bot(
                alpha, distance, grid, model, prev_bot_position, actual_action, possible_actions, rat_probabilities_copy, rat_probabilities, possible_rat_positions
            )
            agreement_log.append(model_agrees)
            metrics_log.append({
                "timestamp": timestamp,
                "actual_action": actual_action,
                "best_action": best_action,
                "agreement": model_agrees,
                "distance": distance,
                "action_scores": action_scores
            })
            # Calculate the score difference
            if actual_action in action_scores and best_action in action_scores:
                score_difference = abs(action_scores[actual_action] - action_scores[best_action])
                action_score_differences.append(score_difference)
            
            # Track all action scores for average computation
            avg_action_score = sum(action_scores.values()) / len(action_scores)
            all_action_scores.append(avg_action_score)

            if model_agrees:
                print(f"Model agrees with the bot's action: {actual_action}")
            else:
                print(f"Model disagrees. Predicted: {best_action}, Actual: {actual_action}")
                print(f"Action Scores: {action_scores}")
        else:
            print("Skipping evaluation ...")

        # Collect data for this iteration
        rat_probabilities_array = np.zeros((GRID_SIZE, GRID_SIZE))
        for pos, prob in rat_probabilities.items():
            rat_probabilities_array[pos] = prob  # Populate the 2D array with probabilities

        # Optional: Add ship layout as a second channel (binary representation)
        ship_layout_array = (grid > 0).astype(float)  # Walls = 1, Open = 0 (binary)


        # Create a grid to represent the bot's position
        bot_position_array = np.zeros((GRID_SIZE, GRID_SIZE))
        bot_position_array[bot_position[0], bot_position[1]] = 1  # Mark bot's position as 1

        # Stack channels to create a tensor of shape [channels x grid_width x grid_height]
        input_tensor = torch.tensor(np.stack([rat_probabilities_array, ship_layout_array, bot_position_array]), dtype=torch.float32)
        data_log.append({
            "input_tensor": input_tensor,  # Input for the model
            "timestamp": timestamp,  # Time elapsed
            "bot_position": bot_position
        })

        # Check if bot has caught the space rat
        if bot_position == rat_position:
            print(f"Bot caught the space rat at {bot_position}", flush=True)
            break
    
    # Calculate average action scores for this simulation
    avg_simulation_action_score = sum(all_action_scores) / len(all_action_scores) if all_action_scores else float('inf')

    # Assign total actions as the target for all logged inputs
    total_actions = state.movements + state.space_rat_detector_actions
    for log in data_log:
        log["total_actions"] = total_actions

    if state.movements >= timeout_limit:
        state.movements = 0
        state.blocked_cell_actions = 0
        state.space_rat_detector_actions = 0
        print("Failed Simulation for catching rat")
        state.failed_simulations += 1

     # Log agreement stats
    if os.path.exists(model_path):
        agreement_percentage = sum(agreement_log) / len(agreement_log) * 100
        print(f"Model-Bot Agreement: {agreement_percentage:.2f}%")

        simulation_metrics = {
            "score_differences": action_score_differences,
            "avg_action_score": avg_simulation_action_score,
            "agreement_percentage": agreement_percentage
        }

    return data_log, simulation_metrics

def run_single_simulation(alpha, data_log, model):
    timeout_limit = 9000
    state = SimulationState()
    state.reset()

    initial_grid = create_space_vessel_layout()
    grid = convert_grid_to_numpy(initial_grid)
    
    bot_position = start_with_initial_cell(grid)

    print("Begin phase 2", flush=True)
    data_log = []
    simulation_metrics = {}
    random_rat_position = get_random_open_position(grid)
    if random_rat_position:
        data_log, simulation_metrics = main_phase_2(grid, bot_position, random_rat_position, alpha, state, data_log, model)
    else:
        print("No cell for space rat", flush=True)

    print("Exit phase 2", flush=True)
    return state.movements, state.blocked_cell_actions, state.space_rat_detector_actions, state.failed_simulations, state.actions, data_log, simulation_metrics

def run_simulation(alpha, data_log, model):
    result = run_single_simulation(alpha, data_log, model)
    return result

def run_simulation_with_timeout(alpha, data_log, model, timeout=30):
    with multiprocessing.Pool(10) as pool:
        async_result = pool.apply_async(run_simulation, (alpha, data_log, model))
        try:
            result = async_result.get(timeout)  # Timeout in seconds
            return result
        except multiprocessing.TimeoutError:
            print(f"Simulation with alpha {alpha} timed out.")
            pool.terminate()
            pool.join()
            return (0, 0, 0, 1, [], [], {})  # Failed simulation placeholder

def plot_loss_over_time(timestamps, losses):
    loss_by_time = {}

    for t, loss in zip(timestamps, losses):
        # Use values directly without .item()
        if t not in loss_by_time:
            loss_by_time[t] = []
        loss_by_time[t].append(loss)

    # Calculate the average loss for each timestamp
    avg_loss_by_time = {t: sum(l) / len(l) for t, l in loss_by_time.items()}
    times, avg_losses = zip(*sorted(avg_loss_by_time.items()))

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(times, avg_losses, marker="o", label="Loss Over Time")
    plt.xlabel("Time Elapsed")
    plt.ylabel("Loss")
    plt.title("Loss as a Function of Time Elapsed")
    plt.legend()
    plt.grid()
    plt.show()

def repeat_losses(train_losses, test_losses, total_timestamps):
    # Calculate how many timestamps each loss value should cover
    repeat_count = total_timestamps // len(train_losses)
    
    # Repeat each loss value for the calculated count
    extended_train_losses = np.repeat(train_losses, repeat_count)
    extended_test_losses = np.repeat(test_losses, repeat_count)
    
    # Ensure the lengths match the total timestamps
    extended_train_losses = extended_train_losses[:total_timestamps]
    extended_test_losses = extended_test_losses[:total_timestamps]

    return extended_train_losses, extended_test_losses

# def plot_loss_comparison_over_time(train_timestamps, train_losses, test_timestamps, test_losses):
#     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     # Group training losses by timestamps
#     train_loss_by_time = {}
#     for t, loss in zip(train_timestamps, train_losses):
#         if t not in train_loss_by_time:
#             train_loss_by_time[t] = []
#         train_loss_by_time[t].append(loss)
#     avg_train_loss_by_time = {t: sum(l) / len(l) for t, l in train_loss_by_time.items()}

#     # Group testing losses by timestamps
#     test_loss_by_time = {}
#     for t, loss in zip(test_timestamps, test_losses):
#         if t not in test_loss_by_time:
#             test_loss_by_time[t] = []
#         test_loss_by_time[t].append(loss)
#     avg_test_loss_by_time = {t: sum(l) / len(l) for t, l in test_loss_by_time.items()}

#     # Sort by timestamps
#     train_times, avg_train_losses = zip(*sorted(avg_train_loss_by_time.items()))
#     test_times, avg_test_losses = zip(*sorted(avg_test_loss_by_time.items()))

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_times, avg_train_losses, marker="o", label="Training Loss Over Time", color="blue")
#     plt.plot(test_times, avg_test_losses, marker="x", label="Testing Loss Over Time", color="orange")
#     plt.xlabel("Time Elapsed")
#     plt.ylabel("Loss")
#     plt.title("Comparison of Training and Testing Losses Over Time Elapsed")
#     plt.legend()
#     plt.grid()

#     # Save the plot
#     filename = f"./plots/loss_comparison_over_time_{current_time}.png"
#     plt.savefig(filename)
#     print(f"Loss Comparison Over Time plot saved as {filename}")
#     plt.close()

def plot_loss_comparison_over_time(train_losses, test_losses):
    # Extend losses
    total_timestamps = 150
    extended_train_losses, extended_test_losses = repeat_losses(train_losses, test_losses, total_timestamps)
    # Plot training and testing losses
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamps = range(1, total_timestamps + 1)
    plt.plot(timestamps, extended_train_losses, label="Training Loss", color="blue")
    plt.plot(timestamps, extended_test_losses, label="Testing Loss", color="orange")
    plt.xlabel("Timestamps")
    plt.ylabel("Loss")
    plt.title("Training and Testing Losses Over Timestamps of a Simulation")
    plt.legend()
    plt.grid()
    filename = f"./plots/loss_comparison_over_time_{current_time}.png"
    plt.savefig(filename)
    print(f"Training and Testing Losses plot saved as {filename}")
    plt.close()
    
def calculate_average_agreement(all_simulations_data):
    # Extract agreement percentages
    agreement_percentages = [sim_data["agreement_percentage"] for sim_data in all_simulations_data]
    
    # Calculate average
    if len(agreement_percentages) > 0:
        avg_agreement = sum(agreement_percentages) / len(agreement_percentages)
    else:
        avg_agreement = 0  # Handle case where there are no simulations
    
    return avg_agreement


def plot_score_differences(best_simulation, avg_agreement):
    # Extract the score differences and timestamps
    score_differences = best_simulation["score_differences"]
    timestamps = range(1, len(score_differences) + 1)  # X-axis: Timestamps (1, 2, 3, ...)

    # Plot training and testing losses
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Plot the score differences
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, score_differences, marker='o', label="Score Difference")
    plt.xlabel("Timestamp")
    plt.ylabel("Score Difference (Actual vs Best Predicted Action)")
    plt.title(f"Score Differences Over Time (Best simulation i.e with Lowest Avg Score across Timestamps) (Avg agreement: {avg_agreement})")
    plt.legend()
    plt.grid(True)
    filename = f"./plots/score_differences_{current_time}.png"
    plt.savefig(filename)
    print(f"Score differences plot saved as {filename}")
    plt.close()

def main():
    INPUT_CHANNELS = 3  # Rat probabilities and ship layout and bot position
    HIDDEN_SIZE = 128
    EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    MODEL_PATH = "space_rat_model.pth"
    alpha = 0.1
    num_simulations = 50

    all_data_logs = []
    all_simulations_data = []

    print(f"Running simulations for alpha {alpha:.1f}", flush=True)

    # Initialize the model
    model = SpaceRatNN(GRID_SIZE, INPUT_CHANNELS, HIDDEN_SIZE)

    # Collect data from simulations
    for _ in range(num_simulations):
        data_log = []
        _, _, _, _, _, data_log, simulation_metrics = run_simulation_with_timeout(alpha, data_log, model)
        all_data_logs.extend(data_log)
        all_simulations_data.append(simulation_metrics)

    if os.path.exists(MODEL_PATH):
        # for sim in all_simulations_data:
        #     if "avg_action_score" not in sim:
        #         print(f"Simulation {sim} does not have an avg action score")
        # Filter out invalid or empty dictionaries
        valid_simulations = [sim for sim in all_simulations_data if sim and "avg_action_score" in sim]
        # Ensure there is at least one valid simulation
        if valid_simulations:
            best_simulation = min(valid_simulations, key=lambda x: x["avg_action_score"])
        else:
            print("No valid simulations found.")
            best_simulation = None  # Or handle it as per your logic

        # Avg agreement percent
        average_agreement = calculate_average_agreement(all_simulations_data)

        # Example usage
        plot_score_differences(best_simulation, average_agreement)

    # Prepare data
    inputs, targets, timestamps = prepare_data_from_logs(all_data_logs, GRID_SIZE)
    train_inputs, test_inputs, train_targets, test_targets, train_timestamps, test_timestamps = split_data(
        inputs, targets, timestamps
    )

    # Reshape inputs for the model (batch x channels x grid_size x grid_size)
    train_inputs = train_inputs.view(-1, INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
    test_inputs = test_inputs.view(-1, INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)

    # Check if a saved model exists
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("Fine-tuning the existing model...")
    else:
        print("Training a new model...")

    # Fine-tune or train the model
    train_losses, test_losses, scaled_train_losses, scaled_test_losses = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, EPOCHS, BATCH_SIZE, LEARNING_RATE
    )

    # Save the fine-tuned model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model fine-tuned and saved to {MODEL_PATH}")

    # Plot training and testing losses
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure()
    # plt.plot(range(EPOCHS), train_losses, label="Training Loss")
    # plt.plot(range(EPOCHS), test_losses, label="Testing Loss")
    plt.plot(range(EPOCHS), scaled_train_losses, label="Training Loss")
    plt.plot(range(EPOCHS), scaled_test_losses, label="Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Testing Loss Over Epochs")
    plt.legend()
    plt.grid()
    filename = f"./plots/training_testing_loss_{current_time}.png"
    plt.savefig(filename)
    print(f"Training and Testing Loss plot saved as {filename}")
    plt.close()

    # Plot loss as a function of time elapsed
    plot_loss_comparison_over_time(scaled_train_losses, scaled_test_losses)


# def main():
#     INPUT_CHANNELS = 2  # Rat probabilities and ship layout
#     HIDDEN_SIZE = 128
#     EPOCHS = 50
#     BATCH_SIZE = 32
#     LEARNING_RATE = 0.001

#     # alphas = np.arange(0.0, 1.1, 0.1)
#     alpha = 0.1
#     num_simulations = 20
#     model_path = "space_rat_model.pth"
#     all_data_logs = []

#     # for alpha in alphas:
#     print(f"Running simulations for alpha {alpha:.1f}", flush=True)

#     for _ in range(num_simulations):
#         data_log = []
#         _, _, _, _, _, data_log = run_simulation_with_timeout(alpha, data_log)
#         all_data_logs.extend(data_log)

#     # Prepare data
#     inputs, targets, timestamps = prepare_data_from_logs(all_data_logs, GRID_SIZE)
#     train_inputs, test_inputs, train_targets, test_targets, train_timestamps, test_timestamps = split_data(
#         inputs, targets, timestamps
#     )

#     # Reshape inputs for the model (batch x channels x grid_size x grid_size)
#     train_inputs = train_inputs.view(-1, INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)
#     test_inputs = test_inputs.view(-1, INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)

#     # Initialize and train the model
#     model = SpaceRatNN(GRID_SIZE, INPUT_CHANNELS, HIDDEN_SIZE)
#     train_losses, test_losses = train_model(
#         model, train_inputs, train_targets, test_inputs, test_targets, EPOCHS, BATCH_SIZE, LEARNING_RATE
#     )

#     # Plot training and testing losses
#     plt.figure()
#     plt.plot(range(EPOCHS), train_losses, label="Training Loss")
#     plt.plot(range(EPOCHS), test_losses, label="Testing Loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training and Testing Loss Over Epochs")
#     plt.legend()
#     plt.show()

#     # Plot loss as a function of time elapsed
#     plot_loss_comparison_over_time(train_timestamps, train_losses, test_timestamps, test_losses)

if __name__ == "__main__":
    main()