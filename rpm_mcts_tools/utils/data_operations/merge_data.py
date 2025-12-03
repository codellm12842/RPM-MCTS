import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils import read_json, write_jsonl


def merge_columns_by_order():
    source_file = '/path/to/input.jsonl'
    target_file = '/path/to/output.jsonl'
    output_file = './merged.jsonl'
    columns_to_copy = ['mcts_think_steps']

    source_data = read_json(source_file)
    target_data = read_json(target_file)

    assert len(source_data) == len(target_data), "Source and target files must have the same number of entries."
    for source_item, target_item in zip(source_data, target_data):
        assert source_item['task_id'] == target_item['task_id'], "Task IDs must match."
        for column in columns_to_copy:
            assert column in source_item, f"Column {column} not found in source data."
            target_item[column] = source_item[column]
    write_jsonl(target_data, output_file)

def merge_columns_by_taskid():
    source_file = '/path/to/input.jsonl'
    target_file = './merged.jsonl'
    output_file = './output.jsonl'
    columns_to_copy = ['difficulty', 'canonical_solution']

    source_data = read_json(source_file)
    target_data = read_json(target_file)
    merge_data = []

    source_dict = {item['task_id']: item for item in source_data}
    for target_item in target_data:
        task_id = target_item['task_id']
        assert task_id in source_dict, f"Task ID {task_id} not found in source data."
        source_item = source_dict[task_id]
        for column in columns_to_copy:
            assert column in source_item, f"Column {column} not found in source data."
            target_item[column] = source_item[column]
        merge_data.append(target_item)
    write_jsonl(merge_data, output_file)

def delete_columns_by_keys():
    input_file = '/path/to/input.jsonl'
    output_file = '/path/to/output.jsonl'
    keys_to_delete = ['is_solved', 'error_message', 'solution', 'mcts_think_steps']

    data = read_json(input_file)
    for item in data:
        for key in keys_to_delete:
            if key in item:
                del item[key]
    write_jsonl(data, output_file)


if __name__ == "__main__":
    # merge_columns_by_order()
    # merge_columns_by_taskid()
    delete_columns_by_keys()