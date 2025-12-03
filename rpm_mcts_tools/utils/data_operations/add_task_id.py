from rpm_mcts_tools.utils.utils import read_json, write_jsonl


def add_task_id(input_path, output_path):
    data = read_json(input_path)

    results = []
    for idx, item in enumerate(data):
        result = {}
        result["task_id"] = idx + 1
        result.update(item)
        results.append(result)
    write_jsonl(results, output_path)

def del_task_id(input_path, output_path, is_sort=True):
    data = read_json(input_path)

    # 是否先按照 task_id 排序
    if is_sort:
        data.sort(key=lambda x: x["task_id"])

    for item in data:
        item.pop("task_id", None)
    write_jsonl(data, output_path)

if __name__ == "__main__":
    input_path = "/path/to/input.jsonl"
    output_path = "/path/to/output.jsonl"

    add_task_id(input_path, output_path)
    # del_task_id(input_path, output_path, is_sort=True)