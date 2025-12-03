import os
import sys
import argparse
import pprint

from rpm_mcts_tools.utils.utils import read_json, extract_output_from_llm_response
from rpm_mcts_tools.utils.chat_models_api import ChatAPI
from rpm_mcts_tools.utils.concurrent_processor import ConcurrentProcessor


sys_prompt = '''problem:
{problem}
canonical analysis
{canonical_steps}
canonical solution:
{canonical_solution}

Your task is to select the most relevant topic from the following topics based on the programming problem and canonical solution.
Note: Analyze first, then output the most relevant topic wrapped in <topic></topic> tags.

topics:
- Data Structures
- Algorithm Strategies
- String Processing
- Sorting and Searching
- Graph Theory
- Bit Manipulation
- Mathematics and Number Theory
- Computational Geometry
- Optimization Problems
- Two-Pointer Techniques
- Dynamic Programming
- Recursion and Backtracking
- Hashing Techniques
- Other
'''

# 定义单条数据处理函数
def process_single_item(item_idx, item, **kwargs):
    task_id = item['task_id']
    problem = item['prompt']
    canonical_steps = item['canonical_steps']
    canonical_solution = item['canonical_solution']
    print(f"Processing task {item_idx + 1}: {task_id}")

    # 获取模型和参数
    model = ChatAPI()
    args = kwargs.get('args')

    # generate code
    prompt = sys_prompt.format(
        problem=problem,
        canonical_steps='\n'.join(canonical_steps),
        canonical_solution=canonical_solution,
    )
    response = model.generate(prompt)[0]
    result = extract_output_from_llm_response(response, tags_to_extract=['topic'])
    topic = result['topic']

    result = {
        "task_id": task_id,
        "topic": topic
    }
    result.update(item)
    return result


def main(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = read_json(args.dataset_path)
    
    # 断点续跑
    if os.path.exists(args.output_path):
        exists_results = read_json(args.output_path)
        exist_ids = {item['task_id'] for item in exists_results}
        print(f"已存在 {len(exist_ids)} 条数据，跳过...")
        data = [item for item in data if item['task_id'] not in exist_ids]
    
    # 如果只运行一条数据
    if args.run_single_example:
        print("测试模式：只运行第一条数据")
        data = data[:1]

    # 使用并行处理器
    processor = ConcurrentProcessor(
        data=data[:],
        process_func=process_single_item,
        output_path=args.output_path,
        max_workers=args.max_workers,
        save_interval=args.save_interval,
        **{"args": args}
    )
    processor.run()

    # 统计不同topic数量
    topic_count = {}
    for item in read_json(args.output_path):
        topic = item['topic']
        if not topic:
            topic = "未分类"
        if topic not in topic_count:
            topic_count[topic] = 0
        topic_count[topic] += 1
    print("Topic统计结果:")
    for topic, count in topic_count.items():
        print(f"{topic}: {count}条数据")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../../output/canonical_steps/processed_canonical_steps_v2.jsonl")
    parser.add_argument("--output_path", type=str, default="../../output/canonical_steps/processed_canonical_steps_with_topic.jsonl")
    parser.add_argument("--max_workers", type=int, default=10, help="并行处理的最大工作线程数")
    parser.add_argument("--save_interval", type=int, default=1, help="每处理多少条数据保存一次结果")
    parser.add_argument("--run_single_example", type=bool, default=False, help="是否只运行一条数据(测试模式)")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)