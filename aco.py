import random
import numpy as np
from fonctions import extraction, calcul_makespan
import optuna

N = 4
data = np.array(extraction("20J-5M.txt")[N][5])
num_machines, num_jobs = data.shape
borne = np.array(extraction("20J-5M.txt")[N][3])
print(data)

pheromones = np.ones((num_jobs, num_jobs))

heuristics = np.zeros((num_jobs, num_jobs))
for i in range(num_jobs):
    for j in range(num_jobs):
        if i != j:
            heuristics[i][j] = 1 / np.sum(data[:, i] + data[:, j])

def construct_solution(alpha, beta):
    job_scores = []
    for job in range(num_jobs):
        total_processing = np.sum(data[:, job])
        pheromone_score = np.sum(pheromones[:, job])
        score = (total_processing ** beta) * (pheromone_score ** alpha)
        job_scores.append((job, score))
    
    sorted_jobs = [job for job, _ in sorted(job_scores, key=lambda x: -x[1])]
    solution = [sorted_jobs[0]]

    for job in sorted_jobs[1:]:
        candidates = []

        for pos in range(len(solution) + 1):
            temp_solution = solution[:pos] + [job] + solution[pos:]
            mks = calcul_makespan(temp_solution, data, num_machines, num_jobs)

            pheromone_influence = 1.0
            if pos > 0:
                pheromone_influence *= pheromones[solution[pos - 1], job]
            if pos < len(solution):
                pheromone_influence *= pheromones[job, solution[pos]]

            weight = (1 / (mks + 1e-6)) ** beta * (pheromone_influence ** alpha)
            candidates.append((temp_solution, weight))

        total = sum(w for _, w in candidates)
        probabilities = [w / total for _, w in candidates]
        selected = random.choices(candidates, weights=probabilities, k=1)[0]
        solution = selected[0]

    return solution

def maj_pheromones(ant_solutions, best_makespan, evaporation_rate, Q):
    global pheromones
    pheromones *= (1 - evaporation_rate)
    
    for solution in ant_solutions:
        solution_makespan = calcul_makespan(solution, data, num_machines, num_jobs)
        if solution_makespan == best_makespan:
            pheromone_deposit = Q / solution_makespan
            #print(f"Meilleur makespan trouvé : {solution_makespan}, dépôt de phéromone maximal : {pheromone_deposit}")
        else:
            pheromone_deposit = Q / (solution_makespan + 1e-10)
        for i in range(len(solution) - 1):
            pheromones[solution[i], solution[i + 1]] += pheromone_deposit

def ACO(alpha, beta, evaporation_rate, num_ants, iterations, Q):
    best_solution = None
    best_makespan = float('inf')
    
    stagnation_window = 30
    no_improve_counter = 0

    for iteration in range(iterations):
        ant_solutions = []
        iteration_best_makespan = float('inf')
        iteration_best_solution = None
        
        for ant in range(num_ants):
            solution = construct_solution(alpha, beta)
            ant_solutions.append(solution)
            makespan = calcul_makespan(solution, data, num_machines, num_jobs)

            if makespan < iteration_best_makespan:
                iteration_best_makespan = makespan
                iteration_best_solution = solution
        
        if iteration_best_makespan < best_makespan:
            best_makespan = iteration_best_makespan
            best_solution = iteration_best_solution
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        maj_pheromones(ant_solutions, best_makespan, evaporation_rate, Q)
        #print(f"Itération {iteration + 1} : {best_makespan}")

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

    global pheromones
    pheromones = np.ones((num_jobs, num_jobs))

    _, makespan = ACO(alpha, beta, evaporation_rate, num_ants, iterations=5, Q=Q)
    return makespan

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=6, n_jobs=6) 

best_params = study.best_params
print("\n-----> Meilleure combinaison trouvée :")
for key, val in best_params.items():
    print(f"{key} = {val}")

pheromones = np.ones((num_jobs, num_jobs)) 
best_solution, best_makespan = ACO(**best_params, iterations=80)

print("Meilleure permutation des jobs:", [job + 1 for job in best_solution])
print("Makespan de la meilleure solution:", best_makespan)
print("RDP = ", (best_makespan - borne) / borne * 100, "% et borne ", borne)