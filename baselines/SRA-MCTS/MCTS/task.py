import re

from MCTS.mcts import MCTS
from prompt.prompts import *
# from models.model import *

def grade_answer(given_answer: str, question: str, lang='en') -> bool:
    if 'Step 2:' not in given_answer:
        return False
    print('-'* 40 + 'The question is' + '-'*40)
    print(question)
    print('-'* 40 + 'The final answer is' + '-'*40)
    print(given_answer)
    if lang == 'en':
        message = completed_prompt_en.format(problem=question, solution=given_answer)
    else:
        message = completed_prompt_zh + "问题: " + question + '\n' + "解决方案: " + given_answer + '\n'
    response = get_proposal(message)
    if '<solved>' in response or '可以解决' in response:
        return True
    elif '<unsolved>' in response or '不能解决' in response:
        return False
    else:
        return False



# def extract_answer(prediction):
#     pattern = r"The final answer is (.*)"
#     match = re.findall(pattern, prediction)
#     if match:
#         # print("match1")
#         answer = match[0]
#     else:
#         if 'answer is' in prediction:
#             answer = prediction.split('answer is')[-1].strip()
#             if len(answer) > 1:
#                 if answer[-1] == '.':
#                     answer = answer[:-1].strip()
#                 if len(answer) > 1:
#                     if answer[0] == ':':
#                         answer = answer[1:].strip()
#         else:
#             answer = prediction
#     return answer


def exact_match_score(prediction, question, lang):
    # prediction = extract_answer(prediction)
    return grade_answer(prediction, question, lang)


# given prompt, generate proposal under instruction, unwrap is required
def get_proposal(prompt, mcts_task, *args, **kwargs):
    response = mcts_task.model.generate(prompt, temperature=0.7)[0]
    print('proposal: \n' + response)
    return response

# given prompt + answer, find its value
# if you use api, unwrap is required. if you use local value model, the value is directly obtained
def get_value(prompt_answer, mcts_task, *args, **kwargs):
    value = mcts_task.value_model.generate(prompt_answer, temperature=0.7)[0]
    print('value: \n' + value)
    return value

# data: question: str
class SearchTask(object):
    def __init__(self, data, propose_method='llama', value_method='llama'):
        super().__init__()
        self.question = data
        self.propose_method = propose_method
        self.value_method = value_method
        self.value_cache = {}

    def clear_cache(self):
        self.value_cache = {}

    @staticmethod
    def summary_prompt_wrap(x: str, y: str = '') -> str:
        print('\n', '==============================', 'summary', '==============================', '\n')
        # print('summary_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤的综述为:\n')
        # prompt = summary_prompt + x + '\n已有步骤:\n' + y + '\n输出:'
        # return prompt
        return x + '\n' + y

    @staticmethod
    def CODE_analyze_summary_prompt_wrap(x: str, y: str = '') -> str:
        print('\n', '==============================', 'CODE_analyze_summary_prompt_wrap', '==============================', '\n')
        # print('summary_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤的综述为:\n')
        prompt = CODE_analyze_summary_prompt + x + '\nSolution: ' + y + '\nExtracted answer:'
        return prompt

    @staticmethod
    def single_propose_prompt_wrap(x: str, y: str = '', step: int = 0) -> str:
        print('\n', '==============================', 'single_propose_prompt_wrap', '==============================', '\nstep: ', step)
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n')
        prompt = code_proposal_prompt_zh + x + '\n已有步骤:\n' + y + '\n输出:'
        return prompt

    @staticmethod
    def zero_single_propose_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh', histories=[]) -> str:
        print('\n', '==============================', 'zero_single_propose_wrap', '==============================', '\nstep: ', step)
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            prompt = code_proposal_prompt_zh + x + '\n已有步骤:\n' + y + '\n输出:'
        else:
            if not y:
                y = 'None\n'
            prompt = code_proposal_prompt_en + x + '\nExisting Steps:\n' + y
            if histories:
                prompt += 'Do not align with the same line of thought as the subsequent content.'
                for idx, history in enumerate(histories):
                    prompt += 'History ' + str(idx) + ": " + history + '.\n'
            prompt += '\nYour output: '
        return prompt

    @staticmethod
    def zero_single_propose_wrap_use_reflection(x: str, y: str = '', step: int = 0, ref: str = '', lang: str = 'zh', histories=[]) -> str:
        print('\n', '==============================', 'zero_single_propose_wrap_use_reflection', '==============================', '\nstep: ', step)
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            if not ref:
                ref = '无\n'
            prompt = code_proposal_prompt_use_reflection_zh + x + '\n已有步骤:\n' + y + '\n意见:' + ref + '\n输出:'
        else:
            if not y:
                y = 'None\n'
            if not ref:
                ref = 'None\n'
            prompt = code_proposal_prompt_use_reflection_en + x + '\nExisting Steps:\n' + y + '\nAnalysis: ' + ref
            if histories:
                prompt += 'Do not align with the same line of thought as the subsequent content.'
                for idx, history in enumerate(histories):
                    prompt += 'History ' + str(idx) + ": " + history + '.\n'
            prompt += '\nYour output: '
        return prompt

    @staticmethod
    def single_reflection_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'single_reflection_wrap', '==============================', '\nstep: ', step)
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            prompt = single_reflection_prompt_zh + x + '\n已有步骤:\n' + y + '\n输出:'
        else:
            if not y:
                y = 'None\n'
            prompt = single_reflection_prompt_en + x + '\nExisting Steps:\n' + y + '\nOutput:'
        return prompt

    @staticmethod
    def single_reflection_wrap_simple(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'single_reflection_wrap_simple', '==============================', '\nstep: ', step)
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤给出的意见:\n')
        if lang == 'zh':
            if not y:
                y = '无\n'
            prompt = single_reflection_prompt_simple + x + '\n已有步骤:\n' + y + '\n输出:'  # simple style
        else:
            if not y:
                y = 'None\n'
            prompt = single_reflection_prompt_simple_en.format(problem=x, steps=y)
            # print(prompt)
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str, lang: str) -> str:
        print('\n', '==============================', 'critic_of_value_prompt_wrap', '==============================', '\n')
        if lang == 'zh':
            value_prompt = critic_zh + x + '\n已有步骤:\n' + y.strip() + '\n输出:'
        else:
            value_prompt = critic_en.format(problem=x.strip(), existing_steps=y.strip())
        return value_prompt

    @staticmethod
    def self_critic_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'self-critic', '==============================', '\n')
        if not y:
            y = 'None\n'
        critic_prompt = self_critic_prompt + x + '\nSolution:\n' + y + '\nScore:'
        return critic_prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, lang:str, low=0, high=10) -> float:
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


class MCTS_Task(SearchTask):
    def __init__(self, data, model, value_model, propose_method='mistral', value_method='mistral', branch=3, end_gate=8.8, roll_policy='greedy',
                 roll_branch=2, roll_forward_steps=1, time_limit=None, iteration_limit=2, exploration_constant=0.5,
                 alpha=0.5, inf=8, temperature=0.7, max_tokens=1024, seed=170, max_length=1024, truncation=True,
                 do_sample=True, max_new_tokens=512, use_case_prompt=False, use_reflection='common', low=0, high=10,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', lang='en', weighted_verify=False):
        super().__init__(data, propose_method, value_method)
        assert 0 <= low < high, "Inappropriate value range!"
        self.mode = 'mcts'
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_length = max_length
        self.truncation = truncation
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.branch = branch
        self.use_case_prompt = use_case_prompt
        self.low = low
        self.high = high
        self.evaluate = evaluate
        self.end_gate = end_gate
        self.use_reflection = use_reflection
        self.roll_policy = roll_policy
        self.roll_branch = roll_branch
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps
        self.alpha = alpha
        self.limit_type = None
        self.INF = inf
        self.node_count = 1
        self.sample_value = sample_value
        self.answer = answer
        self.verify_method = verify_method
        self.reward_model_type = 'vm'
        self.lang = lang
        self.weighted_verify = weighted_verify

        # new add
        self.model = model
        self.value_model = value_model

        # data
        self.question = data['prompt']
        self.given_tests = data['given_tests']
        self.entry_point = data['entry_point']
        self.question = self.question + '\npublic tests: \n' + '\n'.join(self.given_tests)  # 把given_tests拼接到question后面
        

    def update_count(self):
        self.node_count += 1

    def clear_cache(self):
        self.value_cache = {}
        self.node_count = 1

    def set_limit_type(self):
        if self.time_limit is not None:
            if self.iteration_limit is not None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.limit_type = 'time'
        else:
            if self.iteration_limit is None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if self.iteration_limit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.limit_type = 'iterations'

    def extract_proposal(self, p, step_n, y):
        print(p)
        p = re.sub(r'```.*?```', '', p, flags=re.DOTALL)
        if self.lang == 'zh':
            if '下一步:' in p:
                stp = p.split('下一步:')[1].strip()
                if len(stp) < 2:
                    print('输出步骤过短！\n')
                    return ''
                if stp in y:
                    print('输出步骤重复！\n')
                    return ''

                pattern = r'\d\.\s*(.*)'
                match = re.search(pattern, stp)
                if match:
                    stp = match.group(1)

                revised_ = '步骤' + str(step_n) + ':' + stp
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'

            elif '步骤' in p and ':' in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('步骤')[0].strip()
                if len(p_) < 3:
                    print('输出步骤过短！\n')
                    return ''
                if p_[1:] in y:
                    print('输出步骤重复！\n')
                    return ''
                
                pattern = r'\d\.\s*(.*)'
                match = re.search(pattern, p_)
                if match:
                    p_ = match.group(1)

                revised_ = '步骤' + str(step_n) + p_
                print(f'标准化后新的步骤:{revised_}\n')
                return revised_ + '\n'
            else:
                print('输出格式有误！\n')
                return ''

        else:
            if "Next step" in p or 'Next Step' in p:
                # stp = p.split('Next step:')[1].strip()
                match = re.search(r'Next [Ss]tep[:]\s*(.*)', p)
                if match:
                    p = match.group(1).strip()

            elif "Step" in p and ":" in p:
                match = re.search(r'Step \d+:\s*(.*)', p)
                if match:
                    # 提取 ":" 后面的内容
                    p = match.group(1).strip()
                p = re.sub(r'Step \d+:\s*', '', p).strip()
            elif "Analysis:" in p:
                match = re.search(r'Analysis:\s*(.*)', p)
                p = match.group(1).strip()
            if p in y:
                print('输出步骤重复！\n')
                return ''

            pattern = r'\d\.\s*(.*)'
            match = re.search(pattern, p)
            if match:
                p = match.group(1)

            revised_ = 'Step ' + str(step_n) + ': ' + p
            print(f'标准化后新的步骤:{revised_}\n')
            return revised_ + '\n'

    def get_next_step(self, y, step_n, history=[]):
        if self.use_case_prompt:
            prompt = self.single_propose_prompt_wrap(self.question, y, step_n)
        else:
            prompt = self.zero_single_propose_wrap(self.question, y, step_n, self.lang, history)

        response = get_proposal(prompt, self, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('获得下一步失败！\n')
            return ''

        # if len(response) > 5:
        #     response = response[:5]

        # p = ''
        # for _ in response:
        #     p = p + _ + ' '
        # p = p.strip()
        p = response
        p.replace('：', ':')

        return self.extract_proposal(p, step_n, y)


    def extract_reflection(self, p, step_n):

        print('-'*40 + 'In extract_reflection' + '-'*40)
        print(p)
        print('-'*40 + 'Out extract_reflection' + '-'*40)
        if self.lang == 'zh':
            if '已解决' in p or '已经解决' in p or '<end>' in p:
                if step_n > 1:
                    print('此步问题已解决，停止下探。\n')
                    return '<end>'
                else:
                    return '<continue>'

            if self.use_reflection == 'simple':
                return '<continue>'

            if '意见:' not in p:
                print('输出格式有误！\n')
                return ''
            revised_ = p.split('意见:')[1]
            print(f'标准化后的意见:{revised_}\n')
            return revised_

        else:
            # if ' solved' in p or '<end>' in p:
            if '<end>' in p:
                print('标准化后的意见: <end>\n')
                return '<end>'
            else:
                if self.use_reflection == 'simple':
                    return '<continue>'
                if 'Analysis:' not in p:
                    print('输出格式有误！\n')
                    return p
                revised_ = p.split('Analysis:')[1].strip()
                print(f'标准化后的意见:{revised_}\n')
                return revised_

    def get_next_step_use_reflection(self, y, step_n, reflection, history):  # 暂不支持 case-prompt
        propose_prompt = self.zero_single_propose_wrap_use_reflection(self.question, y, step_n, reflection,
                                                                          self.lang)
        response = get_proposal(propose_prompt, self, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, self.max_new_tokens)
        if not response:
            print('获得下一步失败！\n')
            return ''

        print('-'*40 + 'In get_next_step_use_reflection' + '-'*40)
        print(response)
        print('-'*40 + 'Out get_next_step_use_reflection' + '-'*40)

        # if len(response) > 5:
        #     response = response[:5]

        # p = ''
        # for _ in response:
        #     p = p + _ + ' '
        # p = p.strip()
        p = response
        p.replace('：', ':')

        return self.extract_proposal(p, step_n, y)
      
    def get_simple_reflection(self, y, step_n):
        if step_n == 1:
            return '<continue>'
        # if self.lang == 'en':
        #     if 'answer is' in y:
        #         return '<end>'

        reflection_prompt = self.single_reflection_wrap_simple(self.question, y, step_n, self.lang)
        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            cnt -= 1
        if not response:
            print('获得意见失败！\n')
            return '<end>'
                
        print('-'*40 + 'In get_simple_reflection' + '-'*40)
        print(response)
        print('-'*40 + 'Out get_simple_reflection' + '-'*40)

        # p = ''
        # for _ in response:
        #     p = p + _ + ' '
        # p = p.strip()
        p = response

        return self.extract_reflection(p, step_n)
        
    def get_reflection(self, y, step_n):
        if self.lang == 'en':
            if 'answer is' in y or '<end>' in y:
                return '<end>'

        reflection_prompt = self.single_reflection_wrap(self.question, y, step_n, self.lang)

        cnt = 3
        response = []
        while not response and cnt:
            response = get_proposal(reflection_prompt, self, self.propose_method, self.temperature, self.max_tokens,
                                    self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, self.max_new_tokens)
            cnt -= 1
        if not response:
            print('获得意见失败！\n')
            return ''

        # p = ''
        # for _ in response:
        #     p = p + _ + ' '
        # p = p.strip()
        p = response
        return self.extract_reflection(p, step_n)
      
    def get_step_value(self, y):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        
        prompt = self.value_prompt_wrap(self.question, y, self.lang) 
        response = get_value(prompt, self, self.value_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length)
        value = self.value_outputs_unwrap(response, self.lang, self.low, self.high)
        print(f'获评分:{value}\n')
        self.value_cache.update({y: value})
        return value

    def get_summary(self, y):
        if self.lang == 'zh':
            prompt = self.CODE_analyze_summary_prompt_wrap(self.question, y)

            response = get_proposal(prompt, self, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)

            if not response:
                print('获得综述失败！\n')
                return ''
            p = ''
            for _ in response:
                p = p + _ + ' '
            p = p.strip()

            if self.evaluate:
                if len(p) < 1:
                    print('获得综述过短！\n')
                    return ''

                if '综上所述，最终答案是:' not in p:
                    summ = '综上所述，最终答案是:' + p
                    print(f'获得综述:{summ}\n')
                    return summ
                else:
                    summ = '综上所述，最终答案是:' + p.split('综上所述，最终答案是:')[-1]
                    print(f'获得综述:{summ}\n')
                    return summ

            else:
                if len(p) < 1:
                    print('获得综述过短！\n')
                    return ''

                p = p.replace('综上所述,', '综上所述，')
                if '综上所述，' not in p:
                    summ = '综上所述，' + p
                    print(f'获得综述:{summ}\n')
                    return summ
                else:
                    summ = '综上所述，' + p.split('综上所述，')[-1]
                    print(f'获得综述:{summ}\n')
                    return summ

        else: # lang == en
            prompt = self.CODE_analyze_summary_prompt_wrap(self.question, y)
            response = get_proposal(prompt, self, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                    self.max_length,
                                    self.truncation, self.do_sample, 128)
            if not response:
                print('获得综述失败！\n')
                return ''
            p = ''
            for _ in response:
                p = p + _
            summ = p.strip()
            print(f'获得综述:{summ}\n')

            return summ

    def get_CODE_analyze_summary(self, y):
        prompt = self.CODE_analyze_summary_prompt_wrap(self.question, y)
        response = get_proposal(prompt, self, self.propose_method, self.temperature, self.max_tokens, self.seed,
                                self.max_length,
                                self.truncation, self.do_sample, 128)
        if not response:
            print('获得综述失败！\n')
            return ''
        p = ''
        for _ in response:
            p = p + _ + ' '
        p = p.strip()

        print(f'获得综述:{p}\n')
        return p

    # def verify_end_nodes(self, root):
    #     print('-'*40 + 'In verify_end_nodes' + '-'*40)
    #     if self.reward_model_type == 'vm':
    #         end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
    #     else:
    #         end_leaf_nodes = root.get_all_end_root_nodes_prm()
    #     flag = False
    #     for leaf in end_leaf_nodes:
    #         leaf.on_final_route = True
    #         cnt = 5
    #         summ = ''
    #         while cnt:
    #             if self.verify_method == 'string':
    #                 summ = self.get_CODE_analyze_summary(leaf.y)
    #             else:
    #                 summ = self.get_summary(leaf.y)
    #             if summ:
    #                 leaf.summary = summ
    #                 break
    #             else:
    #                 cnt -= 1
    #         if not summ:
    #             summ = extract_summary_from_solution(leaf.y)
    #             leaf.summary = summ

    #         if self.verify_method == 'string':
    #             result = exact_match_score(summ, self.answer)
    #         else:
    #             result = llm_verify(summ, self.answer)
    #         if result:
    #             if self.reward_model_type == 'vm':
    #                 leaf.min_steps_to_correct = 1
    #             else:
    #                 leaf.he = 1
    #             flag = True
    #     return flag, end_leaf_nodes

    # def get_final_solution(self, root, weighted):  # for evaluation
    #     print('-'*40 + 'In get_final_solution' + '-'*40)
    #     if self.reward_model_type == 'vm':
    #         end_leaf_nodes = root.get_all_end_root_nodes_vm(self.end_gate)
    #     else:
    #         end_leaf_nodes = root.get_all_end_root_nodes_prm()

    #     if not end_leaf_nodes or not weighted:
    #         if not end_leaf_nodes:
    #             best_node, best_V = root.getBestV()
    #         else:
    #             sorted_nodes = sorted(end_leaf_nodes, key=lambda x: x.V, reverse=True)
    #             best_node = sorted_nodes[0]
    #         solution = best_node.y
    #         cnt = 5
    #         summ = ''
    #         while cnt:
    #             # summ = self.get_CODE_analyze_summary(solution)
    #             print('-'*40 + "In get_final_solution's if" + '-'*40)
    #             summ = solution
    #             print('-'*40 + "Out get_final_solution's if" + '-'*40)
    #             if summ:
    #                 best_node.summary = summ
    #                 break
    #             else:
    #                 cnt -= 1
    #         if not summ:
    #             summ = extract_summary_from_solution(solution)
    #             best_node.summary = summ
    #         return solution, summ

    #     else:
    #         all_answers = {}  # {answer: [solution, summ, value]}
    #         for leaf in end_leaf_nodes:
    #             cnt = 5
    #             summ = ''
    #             while cnt:
    #                 # summ = self.get_CODE_analyze_summary(leaf.y)

    #                 print('-'*40 + "In get_final_solution's else" + '-'*40)
    #                 summ = solution
    #                 print('-'*40 + "Out get_final_solution's else" + '-'*40)
    #                 if summ:
    #                     leaf.summary = summ
    #                     break
    #                 else:
    #                     cnt -= 1
    #             if not summ:
    #                 summ = extract_summary_from_solution(leaf.y)
    #                 leaf.summary = summ

    #             extracted_answer = extract_answer(summ)
    #             if extracted_answer in all_answers.keys():
    #                 all_answers[extracted_answer][2] += leaf.V
    #             else:
    #                 all_answers[extracted_answer] = [leaf.y, summ, leaf.V]

    #         best_answer = max(all_answers.values(), key=lambda x: x[2])
    #         solution = best_answer[0]
    #         summ = best_answer[1]
    #         return solution, summ

    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, finish, root = MCTS(self)
        # vm
        if self.reward_model_type == 'vm':
            if self.sample_value != 'full':
                solution = node.y
                summ = solution
                print('-'*40 + 'exact match score' + '-'*40)
                # result = exact_match_score(summ, self.question, self.lang)
                result = None
                final_answer = {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish,
                                'accurate': result, 'real_answer': self.answer}

                # MCTS运行完成
                print('-'*40 + 'MCTS finished!' + '-'*40)
                print(f"finish: {finish}")
                print(f"summary: {summ}")
                print(f"result: {result}")

                # 获取所有path
                def get_leaf_nodes(root):
                    leaf_nodes = []
                    def dfs(node):
                        if not node.children:
                            leaf_nodes.append(node)
                        for child in node.children.values():
                            dfs(child)
                    dfs(root)
                    return leaf_nodes
                leaf_nodes = get_leaf_nodes(root)
                print(f"叶节点路径数量: {len(leaf_nodes)}")
                paths = []
                for leaf in leaf_nodes:
                    paths.append({
                        'visit_sequence': leaf.visit_sequence,
                        'is_solution': leaf == node,
                        'value': leaf.V,
                        'steps_num': leaf.depth,
                        'steps': leaf.y,
                    })
                final_answer['paths'] = paths

                return final_answer, root
            # else:
            #     if not self.evaluate:  # generate only
            #         print('-'* 40 + 'NOT SELF Evaluate' + '-'*40)
            #         assert self.answer is not None, 'Answer is None!\n'
            #         flag, end_leaf_nodes = self.verify_end_nodes(root)

            #         # extract policy data
            #         new_policy_samples = []
            #         for leaf in end_leaf_nodes:
            #             solution = leaf.y
            #             summ = leaf.summary
            #             correct = True if leaf.min_steps_to_correct == 1 else False
            #             new_policy_sample = {'solution': solution, 'summary': summ, 'correct': correct}
            #             new_policy_samples.append(new_policy_sample)

            #         # extract value data
            #         if flag:
            #             new_value_samples = root.get_full_value_samples_vm(end_leaf_nodes)
            #         else:
            #             new_value_samples = []
            #         final_answer = {'content': self.question, 'policy_samples': new_policy_samples,
            #                         'value_samples': new_value_samples, 'real_answer': self.answer}
            #         return final_answer, root
            #     else:
            #         assert self.answer is not None, 'Answer is None!\n'
                    
            #         print('-'* 40 + 'Ready to get_final_solution' + '-'*40)
            #         solution, summ = self.get_final_solution(root, self.weighted_verify)
            #         if not summ:
            #             result = False
            #         else:
            #             print('-'* 40 + 'Ready to exact_match_score' + '-'*40)
            #             result = exact_match_score(summ, self.answer)
            #         final_answer = {'content': self.question, 'solution': solution, 'summary': summ, 'finish': finish,
            #                         'accurate': result, 'real_answer': self.answer}
            #         return final_answer, root

        # prm (only sample generation available now)
        # else:
        #     assert self.sample_value, 'Only sampling is supported for prm!\n'
        #     assert self.answer is not None, 'Answer is None!\n'
        #     flag, end_leaf_nodes = self.verify_end_nodes(root)

        #     # extract policy data
        #     new_policy_samples = []
        #     for leaf in end_leaf_nodes:
        #         solution = leaf.y
        #         summ = leaf.summary
        #         correct = True if leaf.he == 1 else False
        #         new_policy_sample = {'solution': solution, 'summary': summ, 'correct': correct}
        #         new_policy_samples.append(new_policy_sample)

        #     # extract value data
        #     if flag:
        #         new_value_samples = root.get_full_value_samples_prm(end_leaf_nodes)
        #     else:
        #         new_value_samples = []
        #     final_answer = {'content': self.question, 'policy_samples': new_policy_samples,
        #                     'value_samples': new_value_samples, 'real_answer': self.answer}
        #     return final_answer, root
