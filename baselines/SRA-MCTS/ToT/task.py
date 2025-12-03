import random
import re
from typing import List, Dict

from ToT.bfs import BFS
from ToT.dfs import DFS
from prompt.prompts import *
from rpm_mcts_tools.utils.utils import extract_python_code


# def get_proposal(prompt, method='llama', temperature=0.7, max_tokens=512, seed=170, max_length=1024, truncation=True,
#                  do_sample=True, max_new_tokens=512):
#     response = []
#     response = local_inference_model(prompt, max_length=max_length, truncation=truncation, do_sample=do_sample,
#                                         max_new_tokens=max_new_tokens, temperature=temperature)
#     # print(response)
#     return response

# def get_value(prompt_answer, method='llama', temperature=0.7, max_tokens=128, seed=170, max_length=1024, low=0, high=10):
#     cnt = 2
#     value = 0
#     while cnt:
#         try:
#             value = local_value_model(prompt_answer, max_length=max_length)
#             break
#         except Exception as e:
#             print(f'获取<{method}>分数失败!\n错误类型:{e}\n')
#             cnt -= 1
#     return value

def get_proposal(prompt, mcts_task):
    response = mcts_task.model.generate(prompt, temperature=0.7)[0]
    print('proposal: \n' + response)
    return response

def get_value(prompt_answer, mcts_task):
    value = mcts_task.value_model.generate(prompt_answer, temperature=0.7)[0]
    print('value: \n' + value)
    return value


class SearchTask(object):
    def __init__(self, data, propose_method='glm', value_method='glm'):
        super().__init__()
        self.question = data['prompt']
        self.given_tests = data['given_tests']
        self.entry_point = data['entry_point']
        self.propose_method = propose_method
        self.value_method = value_method
        self.value_cache = {}

        # 把given_tests拼接到question后面
        self.question = self.question + '\npublic tests: \n' + '\n'.join(self.given_tests)


    def clear_cache(self):
        self.value_cache = {}

    @staticmethod
    def single_propose_prompt_wrap(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        print('propose_prompt: \n', x + '\nExisting Steps:\n' + y + 'Based on the mentioned steps, possible next step is :\n')
        prompt = code_proposal_prompt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'critic', '==============================', '\n')
        value_prompt = critic_en.format(problem=x.strip(), existing_steps=y.strip())
        return value_prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, low=0.0, high=1.0) -> float:
        print('-'*40 + 'In value_outputs_unwrap' + '-'*40)
        print(value_outputs)

        # 依据<score></score>xml标签来提取步骤数
        pattern = r'<score>(.*?)</score>'
        match = re.search(pattern, value_outputs, re.DOTALL)
        try:
            score = float(match.group(1).strip())
        except Exception as e:
            print('提取<score>标签失败')
            score = 0.0

        out_value = min(max(low, score), high)
        print('-'*40 + 'Out value_outputs_unwrap' + '-'*40)
        print(f'获评分:{out_value}\n')
        return out_value

    @staticmethod
    def get_full_step_prompt_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'zero_single_propose_wrap', '==============================', '\nstep: ', step)
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            prompt = code_proposal_prompt_zh + x + '\n已有步骤:\n' + y + '\n输出:'
        else:
            if not y:
                y = 'None\n'
            prompt = code_proposal_prompt_en_2 + x + '\nExisting Steps:\n' + y
            prompt += '\nYour output: '
        return prompt

    @staticmethod
    def extract_step_prompt_wrap(x: str, lang: str = 'zh') -> str:
        print('\n', '==============================', 'extract_step_prompt_wrap', '==============================', '\n')
        if lang == 'zh':
            prompt = extract_step_prompt_zh + x + '\n输出:'
        else:
            prompt = extract_step_prompt_en + x + '\nOutput:'
        return prompt


class ToT_Task(SearchTask):
    def __init__(self, data, model, value_model, propose_method='glm', value_method='glm', algorithm='dfs', branch=3, select_branch=1,
                 max_depth=4, end_gate=8.8, select_method='greedy',
                 temperature=0.7, max_tokens=1024,
                 seed=170, max_length=2048, truncation=True,
                 do_sample=True, max_new_tokens=256, use_case_prompt=False, low=0, high=10, evaluate='', multiply_value=False, lang='en', answer=None, verify_method='string'):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'tot'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.algorithm = algorithm
        self.branch = branch
        self.select_branch = select_branch
        self.max_depth = max_depth
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.select_method = select_method
        self.end_gate = end_gate
        self.node_count = 1
        self.multiply_value = multiply_value
        self.lang = lang
        self.answer = answer
        self.verify_method = verify_method
        # new add
        self.model = model
        self.value_model = value_model

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1


    def extract_proposal(self, p, step_n, y):
        print(p)
        p = re.sub(r'```.*?```', '', p, flags=re.DOTALL)
        if "Next step:" in p or 'Next Step' in p:
            # stp = p.split('Next step:')[1].strip()
            match = re.search(r'Next [Ss]teps?:\s*(.*)', p)
            p = match.group(1).strip()

        elif "Step" in p and ":" in p:
            match = re.search(r'Step \d+:\s*(.*)', p)
            if match:
                # 提取 ":" 后面的内容
                p = match.group(1).strip()
            p = re.sub(r'Step \d+:\s*', '', p).strip()
            
        if len(p) < 2:
            print('输出步骤过短！\n')
            return ''
        if p in y:
            print('输出步骤重复！\n')
            return ''

        pattern = r'\d\.\s*(.*)'
        match = re.findall(pattern, p)
        if match:
            p = re.sub(r'\d\.\s*(.*)', '', p, flags=re.DOTALL).strip() + '\n'
        for _ in match:
            p = p +  _ + '\n'

        revised_ = 'Step ' + str(step_n) + ': ' + p
        print(f'标准化后新的步骤:{revised_}\n')
        return revised_ + '\n'

    def get_next_step(self, y, step_n):
        prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
        

        response = get_proposal(prompt, self)
        if not response:
            print('获得下一步失败！\n')
            return ''

        return self.extract_proposal(response, step_n, y)

    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        prompt = self.value_prompt_wrap(self.question, y)
        response = get_value(prompt, self)
        value = self.value_outputs_unwrap(response, self.low, self.high)
        print(f'获得评分:{value}\n')
        self.value_cache.update({y: value})
        return value
    
    def get_full_step(self, y, step_n):
        prompt = self.get_full_step_prompt_wrap(self.question, y, step_n, self.lang)

        response = get_proposal(prompt, self)
        if not response:
            print('获得下一步失败！\n')
            return '', ''

        # if len(response) > 5:
        #     response = response[:5]

        # p = ''
        # for _ in response:
        #     p = p + _ + ' '
        # p = p.strip()
        p = response
        p.replace('：', ':')

        prompt = self.extract_step_prompt_wrap(p, self.lang)
        response = get_proposal(prompt, self)

        # 依据<step></step>xml标签来提取内容
        full_steps_list = extract_steps_from_xml(response, step_n)
        full_steps = '\n'.join([step['content'] for step in full_steps_list])
        return full_steps

    def run(self):
        self.clear_cache()
        if self.algorithm == 'dfs':
            solution, root, final_node = DFS(self)
        elif self.algorithm == 'bfs':
            solution, root, final_node = BFS(self)
        else:
            print('Unsupported algorithm!\n')
            return {}
        
        # generate code
        prompt = generate_code_prompt_en.format(question=self.question, analysis=solution)
        response = self.model.generate(prompt, temperature=0.7)[0]
        final_code = extract_python_code(response)

        final_answer = {
            'content': self.question,
            'solution': solution,
            'final_code': final_code
        }

        if self.multiply_value:
            multiply_v = final_node.get_multiply_value()
            final_answer.update({'multiply_value': multiply_v})

        return final_answer, root

def extract_steps_from_xml(content: str, step_n: int) -> List[Dict]:
    """
    从包含<step></step>标签的内容中提取步骤信息，并以list[dict]形式返回
    
    Args:
        content: 包含<step></step>标签的字符串
        step_n: 步骤编号

    Returns:
        步骤列表，每个步骤为一个字典
    """
    import re
    
    # 提取所有<step></step>标签中的内容
    step_pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
    steps = step_pattern.findall(content)
    
    # 将提取的步骤转换为字典列表
    result = []
    for i, step_content in enumerate(steps, 1):
        result.append({
            "step_number": i,
            "is_roll": False if i < step_n else True,
            "content": step_content.strip()
        })
    
    return result