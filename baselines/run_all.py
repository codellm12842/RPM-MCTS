import os
import argparse
import pprint
from dotenv import load_dotenv


def set_environment_by_model(model_name):
    if model_name in ["Qwen2.5-7B-Instruct"]:
        os.environ["MODEL_NAME"] = model_name
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
    else:
        raise ValueError(f"未定义的模型名称: {model_name}")

def get_dataset_path(dataset_name):
    if dataset_name == "human-eval-plus":
        return "../../dataset/human-eval-plus/humaneval_plus.jsonl"
    elif dataset_name == "mbpp-plus":
        return "../../dataset/mbpp-plus/mbpp-plus.jsonl"
    elif dataset_name == "code_contests":
        return "../../dataset/code_contests/CodeContests150.jsonl"
    elif dataset_name == "apps_introductory150":
        return "../../dataset/apps_v2/apps_test_introductory150.jsonl"
    elif dataset_name == "apps_interview150":
        return "../../dataset/apps_v2/apps_test_interview150.jsonl"
    elif dataset_name == "apps_competition150":
        return "../../dataset/apps_v2/apps_test_competition150.jsonl"
    else:
        raise ValueError(f"未定义的数据集名称: {dataset_name}")

def main(args):
    # 设置环境变量
    set_environment_by_model(args.base_model)

    # 获取数据集路径
    args.dataset_path = get_dataset_path(args.dataset)

    # 根据method调用
    if args.method == "base-llm":
        from base_llm import base_llm
        base_llm.main(args)
    elif args.method == "SRA-MCTS":
        run_mcts = __import__('SRA-MCTS.run_mcts', fromlist=['main'])
        run_mcts.main(args)
    elif args.method == "ToT":
        run_tot = __import__('SRA-MCTS.run_tot', fromlist=['main'])
        run_tot.main(args)
    elif args.method == "rpm-mcts":
        from RPM_MCTS import run_mcts
        run_mcts.main(args)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('布尔值必须是 yes/no、true/false、t/f、y/n、1/0 中的一个')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --------------------------------------------------------------------------------------------
    parser.add_argument("--base_model", type=str, default="qwen3-235b-a22b")
    parser.add_argument("--method", type=str, default="rpm-mcts")
    parser.add_argument("--dataset", type=str, default="apps_competition150")
    parser.add_argument("--output_code_path", type=str, default="./code.jsonl")
    parser.add_argument("--eval_result_path", type=str, default="./eval.jsonl")
    parser.add_argument("--max_workers", type=int, default=1, help="并行处理的最大工作线程数")
    parser.add_argument("--save_interval", type=int, default=1, help="每处理多少条数据保存一次结果")
    parser.add_argument("--run_single_example", type=str2bool, default=False, help="是否只运行一条数据(测试模式)")
    parser.add_argument("--send_email", type=str2bool, default=False, help="是否发送邮件通知")
    parser.add_argument("--save_intermediate", type=str2bool, default=False, help="是否保存中间结果到单独的文件")
    # --------------------------------------------------------------------------------------------
    parser.add_argument("--rollout", default=5, type=int, help="The maximum number of rollouts.")
    parser.add_argument("--kb_coef", default=0.5, type=float, help="The coefficient for knowledge base.")
    args = parser.parse_args()
    pprint.pprint(vars(args), sort_dicts=False)
    main(args)

    if args.send_email:
        from rpm_mcts_tools.utils.send_email_utils import send_email
        file_name = os.path.basename(args.output_code_path)
        send_email(f"{file_name}任务结束")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        