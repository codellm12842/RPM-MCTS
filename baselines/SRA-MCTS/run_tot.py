import os
import sys
import argparse
import pprint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), './'))
from ToT.task import ToT_Task
from rpm_mcts_tools.utils.utils import read_json
from rpm_mcts_tools.utils.chat_models_api import ChatAPI
from rpm_mcts_tools.utils.concurrent_processor import ConcurrentProcessor
from rpm_mcts_tools.evaluate.evaluate_by_executor import evaluate


# 定义单条数据处理函数
def process_single_item(item_idx, item, **kwargs):
    task_id = item['task_id']
    problem = item['prompt']
    print(f"Processing task {item_idx + 1}: {task_id}")

    # 获取模型和参数
    model = ChatAPI()
    value_model = ChatAPI()
    args = kwargs.get('args')

    # generate mcts_steps
    task = ToT_Task(
        item, model=model, value_model=value_model,
        max_depth=args.rollout, 
        end_gate=10,
    )
    final_answer, root = task.run()

    item['analysis'] = final_answer['solution']
    item["solution"] = final_answer['final_code']
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
    parser.add_argument("--rollout", default=10, type=int, help="The maximum number of rollouts.")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)