import os
import sys
import json
import argparse

from rpm_mcts_tools.executors import HumanevalExecutor
from rpm_mcts_tools.utils.utils import read_json, write_jsonl

"""
Description: The input file should be in JSONL format and must contain the following fields:
- task_id: Task ID
- solution: Function code
- test: Test case code, e.g., "\ndef check(candidate):\n    assert candidate(4, 2, [[2, 5], [3, 7], [1, 3], [4, 0]]) == 3\ncheck(max_of_min_2d_array)\n"
        or a list of test cases, e.g., ["assert max_of_min_2d_array(4, 2, [[2, 5], [3, 7], [1, 3], [4, 0]]) == 3", "assert max_of_min_2d_array(4, 2, [[2, 5], [3, 7], [1, 3], [4, 0]]) == 4"]
"""

def count_solved(results) -> float:
    solved = 0
    count = 0
    for item in results:
        count += 1
        if "is_solved" in item and item["is_solved"]:
            solved += 1
    return float(solved) / count

def count_token_usage(results):
    total_input_tokens = 0
    total_output_tokens = 0
    for item in results:
        if 'token_usage' in item:
            total_input_tokens += item['token_usage'].get('input_token_num', 0)
            total_output_tokens += item['token_usage'].get('output_token_num', 0)
    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

def evaluate(input_path, output_path):
    executor = HumanevalExecutor()
    data = read_json(input_path)

    results = []
    for idx, item in enumerate(data):
        task_id = item['task_id']
        entry_point = item['entry_point']
        solution = item['solution']
        test = item['test']
        # solution = item['canonical_solution']
        # test = item['given_tests']

        is_solved, error_message = executor.evaluate_v2(solution, test, entry_point, timeout=5)
        print(f"task_id:{task_id}", is_solved, error_message)

        new_item = {
            "task_id": task_id,
            "is_solved": is_solved,
            "error_message": error_message,
        }
        item.pop("is_solved", None)
        item.pop("error_message", None)
        new_item.update(item)
        results.append(new_item)

    # save file
    write_jsonl(results, output_path)
    print(f"pass@1: {count_solved(results):.5f}")
    print(f"total token usage: {count_token_usage(results)}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default=f"./debug.jsonl")
    args = parser.parse_args()
    evaluate(args.input_path, args.output_path)
