from .executor_utils import function_with_timeout_process_str, function_with_timeout_process_list
from .executor_utils import function_with_timeout_processpool_no_return
from typing import List, Union
from .executor_types import ExecuteResult, Executor
import sys
sys.set_int_max_str_digits(100000)

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 60) -> ExecuteResult:
        rtns = function_with_timeout_process(func, tests, timeout)
        return ExecuteResult(*rtns)

    def execute_v2(self, func: str, test: Union[str, List[str]], timeout: int = 60) -> ExecuteResult:
        if isinstance(test, str):
            return function_with_timeout_process_str(func, test, timeout)
        else:
            return function_with_timeout_process_list(func, test, timeout)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 10) -> bool:
        code = f"""{func}\n\n{test}\n\ncheck({name})\n"""
        is_solved, timeout = function_with_timeout_processpool_no_return(code, timeout)
        return is_solved, timeout
    
    def evaluate_v2(self, func: str, test: Union[str, List[str]], timeout: int = 10):
        exe_result = self.execute_v2(func, test, timeout)
        return exe_result["is_passing"], exe_result["error_messages"]
