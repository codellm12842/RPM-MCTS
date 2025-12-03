import os
import sys
import argparse
import pprint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), './'))
from MCTS.task import MCTS_Task
from rpm_mcts_tools.utils.utils import *
from rpm_mcts_tools.utils.chat_models_api import ChatAPI
from rpm_mcts_tools.utils.concurrent_processor import ConcurrentProcessor
from rpm_mcts_tools.evaluate.evaluate_by_executor import evaluate


sys_prompt = '''You will play the role of a code implementer, writing a complete code based on the given problem and the step-by-step analysis of the problem. 
Your code must strictly follow the analysis steps provided and should not include your own opinions.
Rules:
- Importing function libraries(like: import math) and output function code only, without main function so that I can call your generated functions directly.
- The output code should be wrapped with code blocks (like ```python). Example: ```python\\ndef add(a, b):\\n    return a + b\\n```.

question:
{question}
analysis:
{analysis}
'''

sys_prompt_v2 = '''You will play the role of a code implementer, writing a complete code based on the given problem and the step-by-step analysis of the problem if the analysis is correct.
If the analysis is incorrect or incomplete, you should correct it and then write the code.
Rules:
- Importing function libraries(like: import math) and output function code only, without main function so that I can call your generated functions directly.
- The output code should be wrapped with code blocks (like ```python). Example: ```python\\ndef add(a, b):\\n    return a + b\\n```.

question:
{question}
analysis:
{analysis}
'''

# 定义单条数据处理函数
def process_single_item(item_idx, item, **kwargs):
    task_id = item['task_id']
    problem = item['prompt']
    print(f"Processing task {item_idx + 1}: {task_id}")

    # 获取模型和参数
    model = ChatAPI()
    value_model = ChatAPI()
    args = kwargs.get('args')

    # 运行mcts
    task = MCTS_Task(
        item, model=model, value_model=value_model, lang='en', 
        iteration_limit=args.rollout, 
        end_gate=10.1,
    )
    final_answer, root = task.run()

    analysis = final_answer['solution']
    finish = final_answer['finish']
    paths = final_answer['paths']

    # 生成代码
    prompt = sys_prompt_v2.format(question=problem, analysis=analysis)
    response = model.generate(prompt, temperature=0.7)[0]
    final_code = extract_python_code(response)

    item['mcts_steps'] = analysis
    item["solution"] = final_code
    item["finish"] = finish
    item["paths"] = paths
    item['token_usage'] = {
        'input_token_num': model.prompt_tokens + value_model.prompt_tokens,
        'output_token_num': model.completion_tokens + value_model.completion_tokens,
    }

    # debug
    # from visualize import visualize
    # visualize(root, task, "", "", "")
    # exit(0)

    return item


def main(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data = read_json(args.dataset_path)
    
    # 断点续跑
    if os.path.exists(args.output_code_path):
        exists_results = read_json(args.output_code_path)
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
        output_path=args.output_code_path,
        max_workers=args.max_workers,
        save_interval=args.save_interval,
        **{"args": args}
    )
    processor.run()
    
    # evaluate
    print("All done! Start evaluating...")
    evaluate(args.output_code_path, args.eval_result_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../../dataset/mbpp-plus/mbpp-plus.jsonl")
    parser.add_argument("--output_code_path", type=str, default="./code.jsonl")
    parser.add_argument("--eval_result_path", type=str, default="./eval.jsonl")
    parser.add_argument("--max_workers", type=int, default=20, help="并行处理的最大工作线程数")
    parser.add_argument("--save_interval", type=int, default=1, help="每处理多少条数据保存一次结果")
    parser.add_argument("--run_single_example", type=bool, default=False, help="是否只运行一条数据(测试模式)")
    # --------------------------------------------------------------------------------------------
    parser.add_argument("--rollout", default=5, type=int, help="The maximum number of rollouts.")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)