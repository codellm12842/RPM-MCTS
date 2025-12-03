import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils import read_json, write_jsonl

def condition(item):
    return item["is_solved"] == False


if __name__ == "__main__":
    input_file = "/path/to/input.jsonl"
    output_file = "/path/to/output.jsonl"
    
    raw_data = read_json(input_file)
    new_data = []
    for item in raw_data:
        if condition(item):
            new_data.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "entry_point": item["entry_point"],
                "test": item["test"],
                "given_tests": [],
                # "canonical_solution": item["canonical_solution"],
                # "difficulty": item["difficulty"],
            })
    write_jsonl(new_data, output_file)
