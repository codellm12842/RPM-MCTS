import os
import json
import argparse
from datetime import datetime

from staticfg import CFGBuilder


def divide(prog):
    try:
        cfg = CFGBuilder().build_from_src('block', prog)
    except Exception as e:
        raise e

    # cfg.build_visual('cfg_ast', 'png')

    prog_lines = prog.split("\n")

    block_insert = set([0] + [block.end() for block in cfg] + [len(prog_lines)])
    block_insert = sorted(list(block_insert))

    code_blocks = []
    for i in range(len(block_insert) - 1):
        code_block = '\n'.join(prog_lines[block_insert[i]: block_insert[i+1]])
        code_blocks.append(code_block)

    return code_blocks


def read_data(file_path):
    with open(file_path, 'r') as f:
        if file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
        print(f"load {file_path} for {len(data)} datas")
        return data


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # read data
    all_data = read_data(args.dataset_path)
    if 'MBPP' in args.dataset_path:
        train_data = [data for data in all_data if data['task_id']>=601 and data['task_id']<=974]

    results = []
    for data in train_data:
        task_id = data['task_id']
        problem = data['prompt']
        code = data['code']

        # divide code into blocks
        divided_blocks = divide(code)

        results.append({
            'task_id': task_id,
            'code': code,
            'blocks': divided_blocks
        })

        a = 1

    # save file
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
        print(f"save to {args.output_path}")


if __name__ == "__main__":

    code = '''def find_Rotations(str):
        tmp = str + str
        n = len(str)
        for i in range(1,n + 1):
            substring = tmp[i: i+n]
            if (str == substring):
                return i
        return n
    '''

    divided_blocks = divide(code)
    print(divided_blocks)

    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_path", type=str, default="../dataset/MBPP/sanitized-mbpp.jsonl")
    # parser.add_argument("--output_path", type=str, default=f"../output/standard_code_blocks.json")
    #
    # args = parser.parse_args()
    # main(args)
