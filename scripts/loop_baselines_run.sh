#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

### 运行参数
# method: 方法，{"base-llm", "sra-mcts", 'rpm-mcts', 'ToT'}
# 注意：批量跑时不要修改对应方法的代码，不要同时运行同一method的不同配置，可以复制整个方法目录并重命名方法名称
# base_model: 基座模型，{"claude37_sonnet", "qwen3-235b-a22b", "qwen3-8b"}
# dataset_name: 数据集名称，{"human-eval-plus", "mbpp-plus", "code_contests", "apps_introductory150", "apps_interview150", "apps_competition150"}
# suffix: 自定义运行后缀，例如 "_v1"
# send_email: 是否发送邮件通知
method="rpm-mcts"
base_model="qwen3-235b-a22b"
suffix="_v1"
max_workers=20
save_intermediate=false


# 定义数据集数组
datasets=("apps_competition150" "apps_interview150" "apps_introductory150" "human-eval-plus" "mbpp-plus" "code_contests")

# 循环处理每个数据集
for dataset_name in "${datasets[@]}"
do
    ### 输出路径(无需修改)
    run_tag="${method}_${base_model}_${dataset_name}${suffix}"
    log_path="../output/logs/${method}/${run_tag}.log"
    mkdir -p "$(dirname "$log_path")"
    output_code_path="../../output/results_code/${method}/${base_model}/${dataset_name}/code_${run_tag}.jsonl"
    eval_result_path="../../output/results_eval/${method}/${base_model}/${dataset_name}/eval_${run_tag}.jsonl"

    ### 推理
    nohup python -u ../baselines/run_all.py \
        --base_model "$base_model" \
        --method "$method" \
        --dataset "$dataset_name" \
        --output_code_path "$output_code_path" \
        --eval_result_path "$eval_result_path" \
        --max_workers $max_workers \
        --save_intermediate $save_intermediate \
        --rollout 5 \
        > "$log_path" 2>&1 &
    
    pid=$!

    ### 控制台输出进程号和相关信息
    echo "-----------------------------------------------------------------------"
    echo "进程已启动，PID: $pid"
    echo "method: $method"
    echo "base_model: $base_model"
    echo "dataset_name: $dataset_name"
    echo "Log path: $log_path"
    echo "-----------------------------------------------------------------------"
    echo "查看进程: ps -ef | grep run_all"

    wait $pid
done

python ../rpm_mcts_tools/utils/send_email_utils.py -m "${run_tag}运行完成"

python ../output/cal_avg_over_all_datasets.py --method "$method" --base_model "$base_model" --suffix "$suffix"
