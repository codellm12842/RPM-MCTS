from threading import Thread
import ast
import astunparse
import multiprocessing
import sys
from io import StringIO


def timeout_handler(_, __):
    raise TimeoutError()

class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        if self.is_alive():
            return None
        return self.ret
    
    def terminate(self):
        self._stop()


def get_call_str(assert_statement: str) -> str:
    '''
    "assert add(1, 2) == 3" -> "add(1, 2)"
    '''
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()


def function_with_timeout(func, args, timeout):
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    try:
        thread = PropagatingThread(target=wrapper)
        thread.start()
        thread.join(timeout)

        if thread.is_alive(): # timeout
            return -1, None
        else: # correctly run
            return 0, result_container[0] # list of sometime
    except Exception as e:
        return -2, e # incorrectly run

def exec_fn_test_ast(code, ast, timeout):
    """
    执行单个assert测试用例

    Args:
        code (str): 代码字符串
        ast (str): 断言字符串
        timeout (int): 超时时间
    Returns:
        tuple: 执行结果，(状态码, ast, 错误信息)
    """
    imports = 'from typing import *\nimport sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n'
    code = f'{imports}\n{code}\n{ast}'

    try:
        rtn = function_with_timeout(exec, (code, globals()), timeout)
    except Exception as e:
        rtn = (-2, e)
    finally:
        if rtn[0] == 0:         # pass
            return 0, ast, None
        elif rtn[0] == -1:      # timeout
            return -1, ast, "TIMEOUT"
        elif rtn[0] == -2:      # error
            return -2, ast, repr(rtn[1])
        
def exec_fn_test_str(code, test, timeout):
    """
    执行字符串形式的测试用例

    Args:
        code (str): 代码字符串
        test (str): 测试字符串
        timeout (int): 超时时间
    Returns:
        tuple: 执行结果，(状态码, 错误信息)
    """
    imports = 'from typing import *\nimport sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n'
    code = f'{imports}\n{code}\n{test}'

    try:
        rtn = function_with_timeout(exec, (code, globals()), timeout)
    except Exception as e:
        rtn = (-2, e)
    finally:
        if rtn[0] == 0:         # pass
            return 0, None
        elif rtn[0] == -1:      # timeout
            return -1, "TIMEOUT"
        elif rtn[0] == -2:      # error
            return -2, repr(rtn[1])

def exec_ast_fn(code, ast, timeout):
    # run code + completed one ast
    # consider timeout
    # extract stdout

    imports = 'from typing import *'
    code = f'{imports}\n{code}\n{ast}'

    try:
        rtn = function_with_timeout(exec, (code, globals()), timeout)
    except Exception as e:
        rtn = (-2, e)
    finally:
        if rtn[0] == -1:
            return rtn[0], ast
        elif rtn[0] == -2: # incorrect result
            return rtn[0], ast
        else:
            return 0, ast

def eval_ast_fn(code, ast, timeout):
    original_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(f"from typing import *\n{code}", globals())
        ast_func_call = get_call_str(ast)
        rtn = function_with_timeout(eval, (ast_func_call, globals()), timeout)

    except Exception as e:
        rtn = (-2, e)
    finally:
        sys.stdout.flush()
        captured_output = sys.stdout.getvalue()
        sys.stdout = original_stdout

        if rtn[0] == -1:
            return "TIMEOUT", captured_output + "\n TIMEOUT", ast
        elif rtn[0] == -2:  # such as OOIndex
            return str(rtn[1]), captured_output + "\n " + str(rtn[1]), ast
        else:  # can run, but wrong results
            return str(rtn[1]), captured_output, ast

def find_syntax_error(code):
    try:
        exec(code)
        return None
    except SyntaxError as e:
        error_message=""
        try:
            error_message = f'  File "{e.filename}", line {e.lineno}\n'
        except:
            pass
        try:
            error_message += f'    {e.text.strip()}\n'
        except:
            pass
        try:
            error_message += ' ' * (e.offset + 3) + '^\n'
        except:
            pass
        try:
            error_message += f"{e.__class__.__name__}: {e.msg}\n"
        except:
            pass
        return error_message
    except Exception as e:
        return None

def function_with_timeout_process(code_str, asserts, timeout):
    result=find_syntax_error(code_str)
    # syntax_error
    if result!=None:
        return False, [f"{asserts[0]} # Real Execution Output: {result}"], None, 0, None, [0], [result]
    with multiprocessing.Pool(processes = multiprocessing.cpu_count() - 2) as pool:
        tasks = [(code_str, ast, timeout) for ast in asserts]
        pool_results = pool.starmap(exec_ast_fn, tasks)
        
        reward = sum(map(lambda x: 1 if x[0] == 0 else 0, pool_results))
        
        # failed_tests_list: the indexes of the assertions that are failed or timeout
        failed_tests_list = [i for i, x in enumerate(pool_results) if x[0] < 0]
        timeout_list = [i for i, x in enumerate(pool_results) if x[0] == -1]

        is_passing = True
        if len(failed_tests_list):
            is_passing = False
        timeout_flag=False
        if len(timeout_list):
            timeout_flag=True

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2) as pool:
        failed_and_timeout_tests = [x[1] for x in pool_results if x[0] < 0]
        tasks = [(code_str, ast, timeout) for ast in failed_and_timeout_tests]
        pool_results = pool.starmap(eval_ast_fn, tasks)
        
        failed_printed_output_list = [x[1] for x in pool_results]
        failed_tests = ["{} # Real Execution Output: {}".format(x[2], x[0]) for x in pool_results]

    return is_passing, failed_tests, None, reward, timeout_flag, failed_tests_list, failed_printed_output_list

def function_with_timeout_process_list(code_str, asserts, timeout):
    with multiprocessing.Pool(processes = 8) as pool:
        tasks = [(code_str, ast, timeout) for ast in asserts]
        pool_results = pool.starmap(exec_fn_test_ast, tasks)  # return List[Tuple]
    
    passed_list, failed_list = [], []
    error_messages = []
    reward = 0
    reward_value = {
        "pass": 10,
        "timeout": 5,
        "assertError": 3,
        "runningError": 1,
    }
    for x in pool_results:
        if x[0] == 0:
            passed_list.append(x)
            reward += reward_value["pass"]
        elif x[0] == -1:
            failed_list.append(x)
            error_messages.append(x[2])
            reward += reward_value["timeout"]
        elif x[0] == -2:
            failed_list.append(x)
            if x[2].startswith("AssertionError"):
                error_messages.append(x[2] + ' # ' + x[1])
                reward += reward_value["assertError"]
            else:
                error_messages.append(x[2])
                reward += reward_value["runningError"]

    is_passing = True if len(passed_list) == len(asserts) else False
    pass_rate = round(len(passed_list) / len(asserts), 3) if len(asserts) > 0 else 1.0
    error_messages = "\n".join(error_messages)
    reward = reward / len(asserts) if len(asserts) > 0 else 0

    return {
        "is_passing": is_passing,
        "pass_rate": pass_rate,
        "reward": reward,
        "error_messages": error_messages,
        "results": pool_results,
    }

def function_with_timeout_process_str(code_str, tests_str, timeout):
    with multiprocessing.Pool(processes = 8) as pool:
        pool_results = pool.starmap(exec_fn_test_str, [(code_str, tests_str, timeout)])  # return List[Tuple]
    
    x = pool_results[0]
    if x[0] == 0:
        is_passing = True
        error_messages = ""
    else:
        is_passing = False
        error_messages = x[1]

    return {
        "is_passing": is_passing,
        "error_messages": error_messages,
        "results": pool_results,
    }

def function_with_timeout_processpool_no_return(code_str, timeout):
    with multiprocessing.Pool(processes = 1) as pool:
        tasks = [(code_str, "", timeout)]
        pool_results = pool.starmap(exec_ast_fn, tasks)
        return pool_results[0][0] == 0, pool_results[0][0]==-1

