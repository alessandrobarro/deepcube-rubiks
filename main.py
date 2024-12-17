import torch
import torch.nn as nn
import torch.optim as optim
from model import CostToGoNet
from cube import RubiksCube
from utils import *
import random
import heapq
import time

# /-------/HYPERPARAMS/-------/
MAX_SCRAMBLES = 7
MIN_SCRAMBLES = 5
SCRAMBLE_INCREMENT = 1
SCRAMBLE_WIN = 100
BATCH_SIZE = 100
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3

#/-------/DATA PREP/-------/
def generate_scrambled_states(num_states, num_scrambles):
    cube = RubiksCube()
    states = []
    scramble_lengths = []
    
    for _ in range(num_states):
        cube.reset_state()
        scramble_moves = random.randint(1, num_scrambles)
        cube.shuffle_state(scramble_moves)
        state = cube.copy_state()
        feature = state_to_feature(state)
        states.append(torch.tensor(feature, dtype=torch.float32))
        scramble_lengths.append(scramble_moves)
    
    return torch.stack(states), torch.tensor(scramble_lengths, dtype=torch.float32)

# /-------/TRAINING/-------/
def train(model, optimizer, loss_fn):
    epoch = 0
    cube = RubiksCube()

    num_scrambles = 1
    SCRAMBLE_INCREMENT = 1

    while epoch < NUM_EPOCHS:
        current_max_scrambles = min(num_scrambles + (epoch // SCRAMBLE_WIN) * SCRAMBLE_INCREMENT, MAX_SCRAMBLES)

        states, lengths = generate_scrambled_states(BATCH_SIZE, current_max_scrambles)
        predictions = model(states).squeeze(-1)

        next_states = []
        for state in states:
            cube.restore_state(feature_to_state(state))
            current_state = cube.copy_state()

            successors = []
            for move in cube.move_keys:
                cube.move(move[0], move[1])
                next_state = cube.copy_state()
                successors.append(next_state) 
                cube.restore_state(current_state)
            
            next_states.append([state_to_feature(next_state) for next_state in successors])
        
        next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for succ in next_states for s in succ])
        next_preds = model(next_states).squeeze(-1)
        next_preds = next_preds.view(BATCH_SIZE, 12)

        targets = 1 + next_preds.min(dim=1)[0]
        
        #print(f"EPOCH {epoch} - MAX SCRAMBLES: {current_max_scrambles}")
        #print(f"TARGETS MIN: {targets.min().item():.4f}, MAX: {targets.max().item():.4f}")
        print(f"PREDICTIONS MIN: {predictions.min().item():.7f}, MAX: {predictions.max().item():.7f}")

        loss = loss_fn(predictions, lengths)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"EPOCH {epoch} - LOSS: {loss.item():.7f}")

        epoch += 1

    return model

# /-------/A* SEARCH/-------/
def a_star_search(cube, model):
    start_state = cube.copy_state()
    start_hash = hash_state(start_state)
    
    open_set = []
    heapq.heappush(open_set, (0, start_hash))
    state_data = {start_hash: (start_state, [], 0)}
    closed_set = set()
    nodes_expanded = 0

    start_time = time.time()
    while open_set:
        _, current_hash = heapq.heappop(open_set)
        current_state, path, g_cost = state_data[current_hash]
        nodes_expanded += 1

        if is_solved(current_state):
            search_time = time.time() - start_time
            return len(path), nodes_expanded, search_time

        closed_set.add(current_hash)


        for move in cube.move_keys:
            cube.restore_state(current_state)
            cube.move(*move)
            next_state = cube.copy_state()
            next_hash = hash_state(next_state)

            if next_hash in closed_set:
                continue

            h_cost = model(torch.tensor(state_to_feature(next_state), dtype=torch.float32)).item()
            f_cost = g_cost + 1 + h_cost

            if next_hash not in state_data or g_cost + 1 < state_data[next_hash][2]:
                state_data[next_hash] = (next_state, path + [move], g_cost + 1)
                heapq.heappush(open_set, (f_cost, next_hash))
            
            #cube.restore_state(current_state) #!!!!!

    return None, nodes_expanded, time.time() - start_time  # no sol found

# /-------/EVAL/-------/
def evaluate_model(model, num_tests=100):
    model.eval()
    cube = RubiksCube()
    total_error = 0.0

    with torch.no_grad():
        for _ in range(num_tests):
            scramble_length = random.randint(MIN_SCRAMBLES, MAX_SCRAMBLES)
            cube.reset_state()
            cube.shuffle_state(scramble_length)

            state = cube.copy_state()
            feature = state_to_feature(state)
            input_tensor = torch.tensor(feature, dtype=torch.float32)

            predicted_cost = model(input_tensor).item()
            error = abs(predicted_cost - scramble_length)
            total_error += error

            print(f"REAL DIST: {scramble_length}, PREDICTED DIST: {predicted_cost:.2f}, ERROR: {error:.2f}")

    avg_error = total_error / num_tests
    print(f"AVERAGE ERROR: {avg_error:.2f}")

# /-------/TEST/-------/
def test(model, num_tests=100):
    total_time = 0
    total_nodes = 0
    total_solution_length = 0

    for _ in range(num_tests):
        cube = RubiksCube()
        cube.shuffle_state(random.randint(MIN_SCRAMBLES, MAX_SCRAMBLES))
        length, nodes, time_elapsed = a_star_search(cube, model)
        print(f'COMPL TEST - LENGTH: {length}, NUM_NODES: {nodes}, ELAPSED_TIME: {time_elapsed}')

        if length is not None:
            total_solution_length += length
            total_nodes += nodes
            total_time += time_elapsed

    avg_length = total_solution_length / num_tests
    avg_nodes = total_nodes / num_tests
    avg_time = total_time / num_tests

    print("/// RESULTS ///")
    print(f"AVG ELAPSED TIME: {avg_time:.4f}")
    print(f"AVG VISITED NODES: {avg_nodes:.2f}")
    print(f"AVG SOL LENGTH: {avg_length:.2f}")

# /-------/EXEC/-------/
if __name__ == "__main__":
    model = CostToGoNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    print("/----------------/TRAINING PHASE/----------------/")
    #model = train(model, optimizer, loss_fn)
    #torch.save(model.state_dict(), 'deepcubea_10001002.pth')

    print("/----------------/EVAL PHASE/----------------/")
    evaluate_model(model)

    print("\n/----------------/TEST PHASE/----------------/")
    model.load_state_dict(torch.load('deepcubea_10001002.pth'))
    model.eval()
    test(model)
