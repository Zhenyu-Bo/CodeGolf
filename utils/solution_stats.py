import os

# input: solution folder
solutions_path = "best_solutions"
n_tasks = 400

tot_codes = 0
tot_score = 0
code_lengths = []

for i in range(1, n_tasks + 1):
    task_id = str(i).zfill(3)
    solution_file = os.path.join(solutions_path, f"task{task_id}.py")
    if os.path.exists(solution_file):
        tot_codes += 1
    else:
        continue
    code_length = os.path.getsize(solution_file)
    tot_score += max(1, 2500 - code_length)
    code_lengths.append(code_length)

# statistics
code_lengths.sort()
print(f"Total codes: {tot_codes}")
print(f"Total score: {tot_score}")
print(f"Code lengths: {code_lengths}")
print(f"Mean code length: {sum(code_lengths) / len(code_lengths)}")

# percentile
print(f"50 percentile: {code_lengths[int(0.5 * len(code_lengths))]}")
print(f"90 percentile: {code_lengths[int(0.9 * len(code_lengths))]}")
print(f"95 percentile: {code_lengths[int(0.95 * len(code_lengths))]}")
print(f"99 percentile: {code_lengths[int(0.99 * len(code_lengths))]}")
