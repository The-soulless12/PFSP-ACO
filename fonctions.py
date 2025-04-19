import numpy as np

def extraction(file_path):
    instances = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("number of jobs"):
            i += 1
            header = list(map(int, lines[i].strip().split()))
            num_jobs, num_machines, seed, upper, lower = header
            i += 1
            if not lines[i].strip().startswith("processing times"):
                raise ValueError(f"Ligne attendue 'processing times :' non trouvée à la ligne {i+1}")
            i += 1
            matrix = []
            for _ in range(num_machines):
                row = list(map(int, lines[i].strip().split()))
                matrix.append(row)
                i += 1
            instances.append((num_jobs, num_machines, seed, upper, lower, matrix))
        else:
            i += 1  
    return instances

def calcul_makespan(permutation, data, num_machines, num_jobs):
    completion_times = np.zeros((num_machines, num_jobs))
    for pos, job in enumerate(permutation):
        for machine in range(num_machines):
            if machine == 0 and pos == 0:
                start_time = 0
            elif machine == 0:
                start_time = completion_times[machine, permutation[pos - 1]]
            elif pos == 0:
                start_time = completion_times[machine - 1, job]
            else:
                start_time = max(completion_times[machine, permutation[pos - 1]], completion_times[machine - 1, job])

            completion_times[machine, job] = start_time + data[machine, job]
    return completion_times[-1, permutation[-1]]