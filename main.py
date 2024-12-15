import torch
import heapq
import time
import numpy as np
from model import initialize_model
from cube import RubiksCube
from utils import *

def get_next_states(cube, current_state):
    """
    Genera tutti gli stati successivi applicando tutte le mosse possibili.
    """
    next_states = []
    for face, direction in cube.move_keys:
        cube.restore_state(current_state)  # Torna allo stato corrente
        original_state = hash_state(cube.state)  # Stato originale come hash
        cube.move(face, direction)  # Applica la mossa
        next_state_feature = state_to_feature(cube.state)

        # Verifica che lo stato sia cambiato
        if original_state == hash_state(cube.state):
            raise ValueError("Lo stato non è stato aggiornato correttamente dopo la mossa.")

        next_states.append(next_state_feature)
    return next_states

def generate_scrambled_states(cube, num_states, max_scramble_moves):
    states = []
    for _ in range(num_states):
        cube.reset_state()
        num_moves = np.random.randint(1, max_scramble_moves + 1)  # Distanza casuale tra 1 e max_scramble_moves
        cube.shuffle_state(num_moves)
        state_feature = state_to_feature(cube.state)  # Usa utils.py
        states.append((state_feature, num_moves))  # Ritorna stato e "distanza" (numero di mosse applicate)
    return states

def train_model(model, optimizer, epochs, batch_size, max_initial_scramble=1, max_scramble_moves=7, increase_interval=100):
    cube = RubiksCube()
    loss_fn = torch.nn.MSELoss()

    # Calcola l'incremento di difficoltà
    #steps = max((max_scramble_moves - max_initial_scramble) // max(1, (epochs // increase_interval)), 1)
    current_scramble_moves = max_initial_scramble

    for epoch in range(epochs):
        # Aumenta la difficoltà ogni 'increase_interval' epoche
        if epoch % increase_interval == 0 and epoch != 0:
            current_scramble_moves = min(current_scramble_moves + 1, max_scramble_moves) # + steps

        # Genera batch di stati
        states = generate_scrambled_states(cube, batch_size, current_scramble_moves)

        inputs = []
        targets = []
        for state_feature, num_moves in states:
            # Simula azioni possibili
            current_state = cube.copy_state()
            successors = get_next_states(cube, current_state)

            # Calcola J'(s) = min_a[1 + J(successor)]
            successor_costs = []
            for successor in successors:
                successor_tensor = torch.tensor(successor, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    successor_costs.append(1 + model(successor_tensor).item())

            # Usa il minimo costo come target
            inputs.append(state_feature)
            targets.append(min(successor_costs))

        # Prepara i dati per PyTorch
        inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # Backward pass e aggiornamento pesi
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Max Scramble Moves: {current_scramble_moves}")

def a_star_search(cube, model, scramble_moves):
    cube.reset_state()
    cube.shuffle_state(scramble_moves)
    initial_state = cube.copy_state()
    initial_hash = hash_state(initial_state)

    # Strutture dati per A*
    open_set = []
    came_from = {}
    g_score = {initial_hash: 0}
    heapq.heappush(open_set, (0, initial_hash))  # (f(x), hash dello stato)

    visited_nodes = 0
    state_map = {initial_hash: initial_state}  # Mappa hash -> stato

    while open_set:
        # Estrai il nodo con il costo più basso
        _, current_hash = heapq.heappop(open_set)
        current_state = state_map[current_hash]
        visited_nodes += 1

        # Controlla se il cubo è risolto
        if is_solved(current_state):
            path = reconstruct_path(came_from, current_hash)
            return len(path) - 1, visited_nodes  # Numero mosse e nodi visitati

        # Genera stati successori
        cube.restore_state(current_state)
        for face, direction in cube.move_keys:
            cube.move(face, direction)
            successor_state = cube.copy_state()
            successor_hash = hash_state(successor_state)

            tentative_g_score = g_score[current_hash] + 1

            if successor_hash not in g_score or tentative_g_score < g_score[successor_hash]:
                g_score[successor_hash] = tentative_g_score

                # Stima euristica h(x) usando il modello
                successor_feature = torch.tensor(
                    state_to_feature(successor_state), dtype=torch.float32
                ).unsqueeze(0)
                h_score = model(successor_feature).item()
                #print(h_score)

                # Aggiungi il successore all'open set
                f_score = tentative_g_score + h_score
                heapq.heappush(open_set, (f_score, successor_hash))
                came_from[successor_hash] = current_hash
                state_map[successor_hash] = successor_state  # Aggiorna mappa hash -> stato

    return None, visited_nodes  # Non risolvibile

def test_model(model, alpha, beta,num_tests=100):
    cube = RubiksCube()
    total_time = 0
    total_moves = 0
    total_nodes = 0

    for i in range(num_tests):
        scramble_moves = np.random.randint(alpha, beta)
        print(f'Test {i+1}: {scramble_moves} mosse')
        start_time = time.time()
        moves, nodes = a_star_search(cube, model, scramble_moves)
        elapsed_time = time.time() - start_time

        if moves is not None:
            total_time += elapsed_time
            total_moves += moves
            total_nodes += nodes

            print(f"{moves} lunghezza, {elapsed_time:.2f}s, {nodes} nodi visitati.")
        else:
            print(f"impossibile risolvere (scramble: {scramble_moves}).")

    print("\n--- Risultati finali ---")
    print(f"Tempo medio: {total_time / num_tests:.2f}s")
    print(f"Lunghezza media soluzione: {total_moves / num_tests:.2f} mosse")
    print(f"Nodi medi visitati: {total_nodes / num_tests:.2f}")

if __name__ == "__main__":
    # Inizializzazione
    model = initialize_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Addestramento
    print("Inizio addestramento...")
    train_model(model, optimizer, epochs=1000, batch_size=1000) # visited_nodes = 100000

    # Salvataggio del modello
    torch.save(model.state_dict(), "deepcubea_model_small.pth")
    print("Modello salvato.")

    # Caricamento del modello per test
    model.load_state_dict(torch.load("deepcubea_model_small.pth")) #small(nn1200,240,res2,ep400,lr1e-4,bs1000)
    model.eval()

    # Testing
    #print("Inizio test...")
    #test_model(model, 5, 8)

