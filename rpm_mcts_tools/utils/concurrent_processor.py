import concurrent.futures
from typing import List, Callable, Any, Dict
from rpm_mcts_tools.utils.utils import write_jsonl_append
from copy import deepcopy
import traceback

class ConcurrentProcessor:
    def __init__(
        self, 
        data: List[Dict[str, Any]], 
        process_func: Callable,
        output_path: str,
        max_workers: int = 5,
        save_interval: int = 100,
        max_retries: int = 1,
        **process_kwargs
    ):
        """
        初始化并行处理器

        Args:
            data: 要处理的数据列表
            process_func: 处理单个项目的函数，接收(item_idx, item)参数
            output_path: 输出结果的保存路径
            max_workers: 并行处理的最大工作线程数
            save_interval: 每处理多少条数据保存一次结果
            max_retries: 失败时最大重试次数
            **process_kwargs: 传递给 process_func 的额外参数
        """
        self.data = data
        self.process_func = process_func
        self.output_path = output_path
        self.max_workers = max_workers
        self.save_interval = save_interval
        self.max_retries = max_retries
        self.process_kwargs = process_kwargs
        
    def run(self) -> List[Dict[str, Any]]:
        """
        开始并行处理数据
        
        Returns:
            List[Dict[str, Any]]: 处理后的数据列表
        """
        print(f"开始并行处理 {len(self.data)} 条数据，使用 {self.max_workers} 个工作线程...")
        processed_results = []
        failed_tasks = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 将任务提交到线程池并记录其索引
            future_to_idx = {}
            for idx, item in enumerate(self.data):
                future = executor.submit(
                    self.process_func, 
                    idx, 
                    item, 
                    **deepcopy(self.process_kwargs) # 使用 deepcopy 确保线程安全
                )
                future_to_idx[future] = idx
            
            # 处理结果
            new_results = []
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result is None: raise ValueError("返回result为None")
                    new_results.append(result)
                    processed_results.append(result)
                    # 中途保存
                    if len(new_results) >= self.save_interval:
                        write_jsonl_append(new_results, self.output_path)
                        new_results = []
                except Exception as e:
                    print(f"执行 {idx} 时发生错误: {e}")
                    traceback.print_exc()
                    failed_tasks.append((idx, self.data[idx]))
        
        # 保存剩余结果
        if new_results:
            write_jsonl_append(new_results, self.output_path)
            
        # 重试失败的任务
        if failed_tasks and self.max_retries > 0:
            print(f"重试 {len(failed_tasks)} 个失败的任务...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as retry_executor:
                # 提交重试任务
                retry_future_to_idx = {}
                for idx, item in failed_tasks:
                    retry_future = retry_executor.submit(
                        self.process_func,
                        idx,
                        item,
                        **deepcopy(self.process_kwargs)
                    )
                    retry_future_to_idx[retry_future] = idx
                
                # 处理重试结果
                retry_results = []
                for retry_future in concurrent.futures.as_completed(retry_future_to_idx):
                    idx = retry_future_to_idx[retry_future]
                    try:
                        result = retry_future.result()
                        if result is None: raise ValueError("重试返回result为None")
                        retry_results.append(result)
                        processed_results.append(result)
                    except Exception as e:
                        print(f"重试 {idx} 时再次失败: {e}")
                        traceback.print_exc()
                
                # 保存重试成功的结果
                if retry_results:
                    write_jsonl_append(retry_results, self.output_path)
                    print(f"重试成功: {len(retry_results)}/{len(failed_tasks)} 个任务")
        
        print(f"所有数据处理完成！共处理 {len(processed_results)} 条数据，失败 {len(self.data) - len(processed_results)} 条。")
        return processed_results