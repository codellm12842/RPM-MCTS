import os
import json
from typing import List, Dict
import pandas as pd
from rpm_mcts_tools.utils.utils import read_json

def check_jsonl_files(file_paths: List[str]) -> Dict:
    """
    检查多个JSONL文件中的数据，确保它们具有相同数量的记录和对应的task_id。
    
    Args:
        file_paths: JSONL文件路径列表
    
    Returns:
        包含所有文件数据的字典，以文件名为键，按task_id排序的数据为值
    """
    # 存储每个文件的数据
    all_data = {}
    
    # 加载所有文件
    for path in file_paths:
        file_name = os.path.basename(path)
        data = read_json(path)
        
        # 确保每个记录都有task_id字段
        if not all(isinstance(item, dict) and 'task_id' in item for item in data):
            print(f"错误: {file_name} 中有记录缺少task_id字段")
            return {}
        
        # 按task_id排序
        sorted_data = sorted(data, key=lambda x: x['task_id'])
        all_data[file_name] = sorted_data
    
    # 检查所有文件的记录数量是否相同
    file_names = list(all_data.keys())
    if not file_names:
        print("没有有效的文件")
        return {}
    
    first_file_len = len(all_data[file_names[0]])
    for name, data in all_data.items():
        if len(data) != first_file_len:
            print(f"错误: 文件记录数量不一致! {file_names[0]}:{first_file_len}, {name}:{len(data)}")
            return {}
    
    # 检查所有文件中的task_id是否一致
    for i in range(1, len(file_names)):
        current_ids = [item['task_id'] for item in all_data[file_names[i]]]
        first_ids = [item['task_id'] for item in all_data[file_names[0]]]
        
        if current_ids != first_ids:
            print(f"错误: {file_names[0]} 和 {file_names[i]} 的task_id不匹配")
            # 找出不匹配的id
            mismatched_ids = set(current_ids).symmetric_difference(set(first_ids))
            if mismatched_ids:
                print(f"不匹配的task_id: {mismatched_ids}")
            return {}
    
    print("✓ 所有检查通过! 文件数量一致且task_id匹配")
    return all_data

def visualize_inconsistent_results(all_data: Dict):
    """
    根据is_solved字段，输出两个不同结果文件中结果不一致的样本。
    
    Args:
        all_data: 包含所有文件数据的字典，以文件名为键，按task_id排序的数据为值
    """
    if len(all_data) != 2:
        print("错误: 需要两个文件进行比较")
        return
    
    # 获取文件名和数据
    file_names = list(all_data.keys())
    data_1 = all_data[file_names[0]]
    data_2 = all_data[file_names[1]]
    
    # 创建一个DataFrame用于存储结果
    inconsistent_results = []
    
    for item_1, item_2 in zip(data_1, data_2):
        if item_1['task_id'] != item_2['task_id']:
            print(f"错误: task_id不匹配 {item_1['task_id']} != {item_2['task_id']}")
            continue
        
        # 比较is_solved字段
        if item_1.get('is_solved') != item_2.get('is_solved'):
            inconsistent_results.append({
                "task_id": item_1['task_id'],
                file_names[0]: "✅" if item_1.get('is_solved') else "❌",
                file_names[1]: "✅" if item_2.get('is_solved') else "❌"
            })
    
    # 如果没有不一致的结果
    if not inconsistent_results:
        print("✓ 所有样本的is_solved字段一致")
        return
    
    # 使用pandas输出表格
    df = pd.DataFrame(inconsistent_results)
    print("以下是结果不一致的样本:")
    print(df.to_string(index=False))

# 使用示例
if __name__ == "__main__":

    input_paths = [
        "/mnt/workspace/codellm/ours/output/results_eval/base-llm/qwen3-235b-a22b/code_contests/eval_base-llm_qwen3-235b-a22b_code_contests_add_given_tests.jsonl",
        "/mnt/workspace/codellm/ours/output/results/sra-mcts/qwen3-235b-a22b/code_contests/eval_sra-mcts_qwen3-235b-a22b_code_contests_kb2_add.jsonl",
    ]

    all_data = check_jsonl_files(input_paths)
    if all_data:
        print("文件检查成功，可以进行下一步的可视化分析")
        visualize_inconsistent_results(all_data)
    else:
        print("文件检查失败，请解决上述问题后再进行可视化")
