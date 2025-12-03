import re
import os
import sys
import argparse
import pprint

from rpm_mcts_tools.utils.utils import read_json
from rpm_mcts_tools.utils.chat_models_api import ChatAPI
from rpm_mcts_tools.utils.concurrent_processor import ConcurrentProcessor


sys_prompt_gen = '''I will provide you with a code problem and its corresponding correct code. Please write down the solution steps for this problem based on this correct code. The steps you analyze will be provided as input to the code large - model for code generation. Please generate your reasoning steps in a way that is easy for the code large - model to understand. Do not include too much content in one step, and make sure the steps are clear and reasonable.
- Let's think step by step. We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
- The entire solution should include at least three steps, so do not skip any necessary steps.
- Your output should be well - organized, describe only one reasoning step at a time, and avoid including multiple reasoning points in a single step.
- Note: You can only describe the reasoning steps in natural language and cannot output code. If you output code, the answer will be discarded.

**Input format:**
Problem:
Correct code:

For each step of thought in the output, first output "Step x". Do not generate any redundant content.
**Output format:**
Step 1:
Step 2:
...
Step n:

Problem:{problem}
Correct code:{canonical_solution}
'''

sys_prompt_extract = '''I'll provide you with a complete answer, and your task is to break down each solution step. Wrap each solution step with the <step></step> XML tag.

**Input format (n steps):**
Complete answer:
Step 1:...
Step 2:...
...
Step n:...
Where "..." denotes omitted input information.

**Output format:**
<step>Step 1:...</step>
<step>Step 2:...</step>
...
<step>Step n:...</step>
Where "..." denotes omitted input information.
The following is the input. Please output according to the specified output format, do not output unnecessary information, and do not repeat the question.
Complete answer:{response}
'''

def extract_steps_from_xml(content):
    # 提取所有<step></step>标签中的内容
    step_pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
    steps = step_pattern.findall(content)

    # 将提取的步骤转换为字典列表
    result = []
    for i, step_content in enumerate(steps, 1):
        result.append({
            "step_number": i,
            "content": step_content.strip()
        })
    
    return result


def process_steps(canonical_steps):
    assert len(canonical_steps) > 0

    full_steps = []
    for i, step in enumerate(canonical_steps, 1):
        content = step['content'].strip()
        assert content, f"Empty content in step {i} for task {task_id}"

        # 如果开头不为Step n，则添加
        if not content.startswith(f"Step {i}:"):
            content = f"Step {i}: {content}"

        # 如果开头为Step n:\n，则将\n换成空格
        if content.startswith(f"Step {i}:\n"):
            content = content.replace(f"Step {i}:\n", f"Step {i}: ")

        full_steps.append(content)

    return full_steps

# 定义单条数据处理函数
def process_single_item(item_idx, item, **kwargs):
    task_id = item['task_id']
    problem = item['prompt']
    canonical_solution = item['canonical_solution']
    print(f"Processing task {item_idx + 1}: {task_id}")

    # 获取模型和参数
    model = ChatAPI()
    args = kwargs.get('args')

    # generate
    prompt1 = sys_prompt_gen.format(problem=problem, canonical_solution=canonical_solution)
    response1 = model.generate(prompt1)[0]
    
    # post-process
    prompt2 = sys_prompt_extract.format(response=response1)
    response2 = model.generate(prompt2)[0]
    canonical_steps = extract_steps_from_xml(response2)
    canonical_steps = process_steps(canonical_steps)

    item["canonical_steps"] = canonical_steps
    return item


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./input.jsonl")
    parser.add_argument("--output_path", type=str, default="./output.jsonl")
    parser.add_argument("--max_workers", type=int, default=1, help="并行处理的最大工作线程数")
    parser.add_argument("--save_interval", type=int, default=1, help="每处理多少条数据保存一次结果")
    parser.add_argument("--run_single_example", type=bool, default=False, help="是否只运行一条数据(测试模式)")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)