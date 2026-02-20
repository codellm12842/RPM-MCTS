import os
import argparse

from rpm_mcts_tools.utils.utils import read_json
from rpm_mcts_tools.evaluate.evaluate_by_executor import count_solved

dataset_nums = {
    "apps_introductory150": 150,
    "apps_interview150": 150,
    "apps_competition150": 150,
    "code_contests": 150,
    "human-eval-plus": 164,
    "mbpp-plus": 378,
}
def main(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    all_results = []
    avg_score = 0
    sum_samples = 0
    for dataset_name, num in dataset_nums.items():
        run_tag = f"{args.method}_{args.base_model}_{dataset_name}{args.suffix}"
        eval_result_path = f"./results_eval/{args.method}/{args.base_model}/{dataset_name}/eval_{run_tag}.jsonl"
        results = read_json(eval_result_path)
        samples = len(results)
        pass1 = count_solved(results)
        all_results.append([dataset_name, samples, pass1])
        avg_score += pass1 * samples
        sum_samples += samples

    # 打印表格
    print("{:<25} {:<10} {:<10}".format("Dataset", "Samples", "Pass@1"))
    for dataset_name, samples, pass1 in all_results:
        print("{:<25} {:<10} {:<10}".format(dataset_name, f"{samples == dataset_nums[dataset_name]}({samples})", f"{pass1:.5f}"))

    # 计算加权平均分
    print("Average score: {:.5f}".format(avg_score / sum_samples))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="qwen3-235b-a22b")
    parser.add_argument("--method", type=str, default="rpm-mcts_roll_add_kb")
    parser.add_argument("--suffix", type=str, default="_roll_add_kb")
    args = parser.parse_args()
    main(args)