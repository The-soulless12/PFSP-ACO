import random
import numpy as np
from fonctions import extraction, calcul_makespan
import optuna

N = 1
data = np.array(extraction("20J-5M.txt")[N][5])
num_machines, num_jobs = data.shape
borne = np.array(extraction("20J-5M.txt")[N][3])
print(data)

def distance(i, j, processing_times):
    m = processing_times.shape[0]  # nombre de machines
    max_delay = 0
    for k in range(2, m):
        delay = np.sum(processing_times[k, i]) - np.sum(processing_times[k - 1, j])
        if delay > max_delay:
            max_delay = delay
    return processing_times[0, j] + max(0, max_delay)

def initialize_heuristics(processing_times):
    num_jobs = processing_times.shape[1]
    heuristics = np.zeros((num_jobs, num_jobs))
    for i in range(num_jobs):
        for j in range(num_jobs):
            if i != j:
                heuristics[i][j] = 1 / distance(i, j, processing_times)
    return heuristics

def construct_solution(alpha, beta, pheromones, heuristics):
    current_job = random.randint(0, num_jobs - 1)
    solution = [current_job]

    while len(solution) < num_jobs:
        unscheduled = [j for j in range(num_jobs) if j not in solution]
        probabilities = []
        for job in unscheduled:
            tau = pheromones[current_job][job] ** alpha
            eta = heuristics[current_job][job] ** beta
            probabilities.append(tau * eta)

        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        next_job = random.choices(unscheduled, weights=probabilities, k=1)[0]
        solution.append(next_job)
        current_job = next_job

    return solution

def maj_pheromones(pheromones, ant_solutions, evaporation_rate, Q):
    pheromones *= (1 - evaporation_rate)
    for solution, mks in ant_solutions:
        for i in range(len(solution) - 1):
            pheromones[solution[i], solution[i + 1]] += Q / (mks + 1e-10)

def ACO(alpha, beta, evaporation_rate, num_ants, iterations, Q):
    pheromones = np.ones((num_jobs, num_jobs))
    heuristics = initialize_heuristics(data)

    best_solution = None
    best_makespan = float('inf')
    stagnation_window = 10000
    no_improve_counter = 0

    for iteration in range(iterations):
        ant_solutions = []
        iteration_best_makespan = float('inf')
        iteration_best_solution = None

        for _ in range(num_ants):
            solution = construct_solution(alpha, beta, pheromones, heuristics)
            mks = calcul_makespan(solution, data, num_machines, num_jobs)
            ant_solutions.append((solution, mks))

            if mks < iteration_best_makespan:
                iteration_best_makespan = mks
                iteration_best_solution = solution

        if iteration_best_makespan < best_makespan:
            best_makespan = iteration_best_makespan
            best_solution = iteration_best_solution
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        maj_pheromones(pheromones, ant_solutions, evaporation_rate, Q)

        if no_improve_counter >= stagnation_window:
            print(f"--| Arrêt anticipé à l'itération {iteration + 1} (aucune amélioration depuis {stagnation_window} itérations)")
            break

    return best_solution, best_makespan

def objective(trial):
    alpha = trial.suggest_float('alpha', 1, 5.0)
    beta = trial.suggest_float('beta', 2, 8.0)
    evaporation_rate = trial.suggest_float('evaporation_rate', 0.05, 0.4)
    Q = trial.suggest_float('Q', 1, 5)
    num_ants = trial.suggest_int('num_ants', 6, 15)

    _, makespan = ACO(alpha, beta, evaporation_rate, num_ants, iterations=5, Q=Q)
    return makespan

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=6, n_jobs=6)

best_params = study.best_params
print("\n-----> Meilleure combinaison trouvée :")
for key, val in best_params.items():
    print(f"{key} = {val}")

best_solution, best_makespan = ACO(**best_params, iterations=300)

print("Meilleure permutation des jobs:", [job + 1 for job in best_solution])
print("Makespan de la meilleure solution:", best_makespan)
print("RDP =", (best_makespan - borne) / borne * 100, "% et borne", borne)
