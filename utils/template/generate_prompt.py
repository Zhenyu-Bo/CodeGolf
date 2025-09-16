import os
import json
from templates import vary_template

def read_code(filepath):
    with open(filepath, encoding='utf-8') as f:
        return f.read().strip()
    
def extract_generate_function(code):
    """Extract the generate function from the code."""
    return "def " + code.split("def")[1].strip()

def read_examples(json_path):
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    examples_str = ""
    max_len = 12000
    cur_len = 0
    num = 0
    for split in ['train', 'test', 'arc-gen']:
        for ex in data.get(split, []):
            inp = ex['input']
            out = ex['output']
            cur_len += len(str(inp)) + len(str(out))
            num += 1
            if cur_len > max_len or num > 50:
                return examples_str
            examples_str += f"Example {num}:\ninput grid:\n{inp}\noutput grid:\n{out}\n"
    return examples_str

def fill_template(task_id):
    # 路径拼接
    gen_path = f'../../generate/task{task_id:03d}.py'
    sol_path = f'../../best_solutions/task{task_id:03d}.py'
    data_path = f'../../google-code-golf-2025/task{task_id:03d}.json'
    out_path = f"../../files/filled_prompt_task{task_id:03d}.txt"

    # 读取内容
    generate_code = extract_generate_function(read_code(gen_path))
    p_code = read_code(sol_path)
    examples_str = read_examples(data_path)

    filled_template = vary_template.format(generate_code=generate_code,p_code=p_code,examples_str=examples_str)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(filled_template)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=1)
    args = parser.parse_args()
    fill_template(args.task_id)