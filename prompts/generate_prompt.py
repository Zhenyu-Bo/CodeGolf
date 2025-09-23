import json
from templates import vary_template

def read_code(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read().strip()
    
def extract_generate_function(code):
    return "def " + code.split("def")[1].strip()

def read_examples(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    examples_str = ""
    max_len = 12000
    cur_len = 0
    num = 0
    for split in ['train', 'test', 'arc-gen']:
        for ex in data.get(split, []):
            inp = ex['input']
            out = ex['output']
            cur_len += len(inp) + len(out)
            num += 1
            if cur_len > max_len or num > 10:
                return examples_str
            examples_str += f"Example {num}:\nInput grid: {inp}\nOutput grid: {out}\n\n"
    return examples_str

def fill_template(task_id):
    gen_path = f'../generate/task{task_id:03d}.py'
    sol_path = f'../solutions/latest/task{task_id:03d}.py'
    data_path = f'../google-code-golf-2025/task{task_id:03d}.json'
    out_path = f'filled_prompt_task{task_id:03d}.txt'
    
    gen_code = extract_generate_function(read_code(gen_path))
    p_code = read_code(sol_path)
    examples_str = read_examples(data_path)

    filled_template = vary_template.format(
        generate_code=gen_code,
        p_code=p_code,
        examples_str=examples_str
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(filled_template)
        print(f"Successfully generate prompt of task{task_id:03d}, save to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, help='Task ID to process', default=1)
    args = parser.parse_args()
    fill_template(args.task)