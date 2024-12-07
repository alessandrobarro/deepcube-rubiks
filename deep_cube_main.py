import torch
import heapq
import random
import time

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from cube import RubiksCube
from utils import *
from deep_cube_model import DeepCubeAModel
import torch
import torch.nn as nn
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf)

#-----------------------Graph and state generation-----------------------#
def update_values(model, cube, optimizer, max_scramble_length, batch_size=64):
    model.train()
    optimizer.zero_grad()

    # Generate training states
    features, labels = generate_training_data(cube, batch_size, max_scramble_length)

    # Convert features and labels to PyTorch tensors
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Ensure labels have the correct shape

    # Predict cost-to-go for current states
    current_values = model(features)

    # Generate next states and compute cost-to-go
    next_features = []
    for feature in features.numpy():  # Convert back to NumPy for compatibility with cube functions
        state = feature_to_state(feature)
        for move in cube.move_keys:
            cube.restore_state(state)
            cube.move(move[0], move[1])
            next_features.append(state_to_feature(cube.copy_state()))

    next_features = torch.tensor(np.array(next_features), dtype=torch.float32)
    with torch.no_grad():
        next_values = model(next_features)

    # Compute target values
    target_values = 1 + next_values.view(-1, len(cube.move_keys)).min(dim=1, keepdim=True).values

    # Compute loss (Mean Squared Error)
    loss = nn.MSELoss()(current_values, target_values)

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item()

#---------------------Dataset construction---------------------#
def generate_training_data(cube, num_samples, max_scramble_length):
    features = []
    labels = []
    
    for _ in range(num_samples):
        # Reset to the solved state
        cube.reset_state()

        # Randomly choose the number of moves to scramble
        scramble_length = random.randint(1, max_scramble_length)
        scramble_moves = random.choices(cube.move_keys, k=scramble_length)

        # Apply the scramble
        for move in scramble_moves:
            cube.move(move[0], move[1])
        
        # Save the scrambled state's features
        scrambled_state = cube.copy_state()
        scrambled_features = state_to_feature(scrambled_state)
        features.append(scrambled_features)

        # Label the state with the cost-to-go
        labels.append(scramble_length)

    return np.array(features), np.array(labels)

#------------------------Train and Eval------------------------#
def train_deepcubea(model, cube, optimizer, num_epochs, max_initial_scramble=1, max_final_scramble=10, batch_size=64):
    max_scramble_length = max_initial_scramble
    for epoch in range(num_epochs):

        if epoch % 30 == 0 and max_scramble_length < max_final_scramble:
            max_scramble_length += 1

        loss = update_values(model, cube, optimizer, max_scramble_length, batch_size)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}, Max Scramble: {max_scramble_length}")

#---------------------Prediction and Search--------------------#
def a_star_search(cube, model, max_moves=10000):
    open_set = []
    heapq.heappush(open_set, (0, 0, hash_state(cube.copy_state()), []))  # Usa la rappresentazione hashabile
    nodes_visited = 0
    closed_set = set()

    while open_set:
        f_score, g_score, hashable_state, moves = heapq.heappop(open_set)
        current_state = state_from_hash(hashable_state)  # Riconversione
        nodes_visited += 1

        if cube.is_solved(current_state):
            return moves, nodes_visited

        if len(moves) >= max_moves:
            continue

        closed_set.add(hashable_state)

        for move in cube.move_keys:
            neighbor_cube = RubiksCube()
            neighbor_cube.restore_state(current_state)
            neighbor_cube.move(move[0], move[1])

            neighbor_state = neighbor_cube.copy_state()
            hashable_neighbor = hash_state(neighbor_state)

            if hashable_neighbor in closed_set:
                continue

            features = torch.tensor(np.array([state_to_feature(neighbor_state)]), dtype=torch.float32)
            with torch.no_grad():
                h = model(features).item()

            f = 0.1 * (g_score + 1) + h # g_score + 1 + h
            print(f)
            heapq.heappush(open_set, (f, g_score + 1, hashable_neighbor, moves + [move]))

    return None, nodes_visited

def retrieve_action_moves(graph, solution_path):
    action_moves = []
    
    for i in range(len(solution_path) - 1):
        current_node = solution_path[i]
        next_node = solution_path[i + 1]
        
        edge_data = graph.get_edge_data(current_node, next_node)
        
        if edge_data is not None:
            action = edge_data['action']
            action_tuple = (action[0], action[1])
            action_moves.append(action_tuple)
        else:
            action_moves.append(None)
    
    return action_moves

#-----------------------------Plot------------------------------#
def plot_graph(graph):
    pos = nx.spring_layout(graph, k=0.5)
    node_sizes = [5] * len(graph.nodes())
    node_sizes[0] = 7
    node_colors = ['lightblue'] * len(graph.nodes())
    node_colors[0] = 'lime'

    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, label=False)
    nx.draw_networkx_edges(graph, pos, edge_color='grey', arrows=False)
    plt.show()

if __name__ == "__main__":

    # Hyperparameters
    input_size = 48
    hidden_size = 1000 #1000
    num_residual_blocks = 4
    num_epochs = 300
    batch_size = 5000 #10000
    max_initial_scramble = 1
    max_final_scramble = 10 #30

    # Initialize cube and model
    cube = RubiksCube()
    model = DeepCubeAModel(input_size=input_size, hidden_size=hidden_size, num_residual_blocks=num_residual_blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


    # Train the model
    print("Training DeepCubeA...")
    train_deepcubea(model, cube, optimizer, num_epochs, max_initial_scramble, max_final_scramble, batch_size)
    torch.save(model.state_dict(), "deepcubea_model.pth")
    print("Model trained and saved.")
    #model.load_state_dict(torch.load("deepcubea_model.pth"))
    #print("Model loaded.")

    # Load model for testing
    model.eval()
    print("Testing DeepCubeA...")

    comptime = []
    lengths = []
    numnodes = []

    for test_iter in range(100):  # Number of test configurations
        # Generate a random scrambled state
        nshuffle = random.randint(1, 3)
        test_cube = RubiksCube()
        test_cube.shuffle_state(nshuffle)
        print(f"Test configuration generated from {nshuffle} random moves")

        # Solve the cube using A* search
        start_time = time.time()
        solution_moves, nodes_visited = a_star_search(test_cube, model)
        end_time = time.time()

        # Collect metrics
        if solution_moves:
            print(f"Solution found in {len(solution_moves)} moves")
        else:
            print("No solution found")

        elapsed_time = end_time - start_time
        comptime.append(elapsed_time)
        lengths.append(len(solution_moves) if solution_moves else 0)
        numnodes.append(nodes_visited)

        print(f"Execution time: {elapsed_time:.4f} seconds")
        print(f"Nodes visited: {nodes_visited}")

    # Print overall metrics
    avg_time = sum(comptime) / len(comptime)
    avg_length = sum(lengths) / len(lengths)
    avg_nodes = sum(numnodes) / len(numnodes)

    print(f"Average execution time: {avg_time:.4f} seconds")
    print(f"Average solution length: {avg_length:.4f} moves")
    print(f"Average nodes visited: {avg_nodes:.4f}")
