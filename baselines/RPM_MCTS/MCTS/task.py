import re
import json
from typing import List, Dict
from threading import Lock
model_load_lock = Lock()

from MCTS.mcts import MCTS
from prompt.prompts import *
from models.embedding_model import filter_similar_sentences
from rpm_mcts_tools.executors.HumanevalExecutor import HumanevalExecutor
from rpm_mcts_tools.utils.utils import extract_python_code, extract_output_from_llm_response
from rpm_mcts_tools.knowledge_base.vector_db_search import DocumentSearch


def get_proposal(prompt, mcts_task, *args, **kwargs):
    response = mcts_task.model.generate(prompt, temperature=0.7)[0]
    print('proposal: \n' + response)
    return response


def get_value(prompt_answer, mcts_task, *args, **kwargs):
    value = mcts_task.value_model.generate(prompt_answer, temperature=0.7)[0]
    print('value: \n' + value)
    return value


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


class SearchTask:
    @staticmethod
    def get_full_step_prompt_wrap(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'zero_single_propose_wrap', '==============================', '\nstep: ', step)
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n')
        if lang == 'zh':
            raise NotImplementedError("Chinese is not yet implemented.")
        else:
            if not y:
                y = 'None\n'
            prompt = code_proposal_prompt_en_v2 + x + '\nExisting Steps:\n' + y
            prompt += '\nYour output: '
        return prompt
    
    @staticmethod
    def extract_step_prompt_wrap(x: str, lang: str = 'zh') -> str:
        print('\n', '==============================', 'extract_step_prompt_wrap', '==============================', '\n')
        if lang == 'zh':
            raise NotImplementedError("Chinese is not yet implemented.")
        else:
            prompt = extract_step_prompt_en + x + '\nOutput:'
        return prompt
    
    @staticmethod
    def zero_single_propose_wrap_use_reflection(x: str, y: str = '', step: int = 0, ref: str = '', lang: str = 'zh', histories=[]) -> str:
        print('\n', '==============================', 'zero_single_propose_wrap_use_reflection', '==============================', '\nstep: ', step)
        # print('propose_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤，可能的当前步骤解法是:\n')
        if lang == 'zh':
            raise NotImplementedError("Chinese is not yet implemented.")
        else:
            if not y:
                y = 'None\n'
            if not ref:
                ref = 'None\n'
            prompt = code_proposal_prompt_use_reflection_en + x + '\nExisting Steps:\n' + y + '\nAnalysis: ' + ref
            if histories:
                # prompt += 'Do not align with the same line of thought as the subsequent content.'
                prompt += '\nThe historical content is the solution proposed earlier. To ensure the diversity of solutions, please do not generate ideas identical to those in the historical content.\n'
                for idx, history in enumerate(histories):
                    prompt += 'History ' + str(idx) + ": " + history + '.\n'

            prompt += 'Your response should only generate solutions to the problem, without any extra words.\n'
            prompt += '\nYour output: '
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str, lang: str) -> str:
        print('\n', '==============================', 'critic_of_value_prompt_wrap', '==============================', '\n')
        if lang == 'zh':
            raise NotImplementedError("Chinese is not yet implemented.")
        else:
            value_prompt = critic_en.format(problem=x.strip(), existing_steps=y.strip())
        return value_prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, lang: str, low=0, high=10) -> float:
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
    def rollout_full_step_wrap(
        problem: str,
        solution: str,
        code: str,
        sandbox_result: str,
        step: int = 0, lang: str = 'zh',
    ) -> str:
        print('\n', '==============================', 'rollout_full_step_wrap', '==============================', '\nstep: ', step)
        if lang == 'zh':
            raise NotImplementedError("Chinese is not yet implemented.")
        else:
            prompt = rollout_full_steps_prompt_en.format(
                problem=problem,
                solution=solution,
                code=code,
                sandbox_result=sandbox_result,
            )
        return prompt
    
    @staticmethod
    def zero_single_propose_wrap_use_knowledgebase(
        x: str, lang: str, content: str, histories=[]
    ) -> str:
        print('\n', '==============================', 'zero_single_propose_wrap_use_knowledgebase', '==============================')
        if lang == 'zh':
            raise NotImplementedError("Chinese is not yet implemented.")
        else:
            prompt = code_proposal_prompt_use_knowledgebase_en.format(problem=x, rag_content=content)

            if histories:
                # prompt += 'Do not align with the same line of thought as the subsequent content.'
                prompt += '\nThe historical content is the solution proposed earlier. To ensure the diversity of solutions, please do not generate ideas identical to those in the historical content.\n'
                for idx, history in enumerate(histories):
                    prompt += 'History ' + str(idx) + ": " + history + '.\n'

            prompt += 'Your response should only generate solutions to the problem, without any extra words.\n'
            prompt += '\nYour output: '
        return prompt


class MCTS_Task(SearchTask):
    def __init__(self, data, model, value_model, logger, propose_method='mistral', value_method='mistral', branch=3, end_gate=9.0, roll_policy='fullsteps',
                 roll_branch=2, roll_forward_steps=1, time_limit=None, iteration_limit=2, exploration_constant=0.5,
                 alpha=0.5, inf=8, temperature=0.7, max_tokens=1024, seed=170, max_length=1024, truncation=True,
                 do_sample=True, max_new_tokens=512, use_case_prompt=False, use_reflection='common', low=0, high=10,
                 evaluate='', sample_value='simple', answer=None, verify_method='string', lang='en', weighted_verify=False,
                 use_embedding_diversity=True, use_knowledge_base1=False, use_knowledge_base2=False, kb_coef=0.5, use_ldb_debug=True):
        super().__init__()
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
        self.roll_branch = roll_branch    # Deprecated
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.exploration_constant = exploration_constant
        self.roll_forward_steps = roll_forward_steps    # Deprecated
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
        self.value_cache = {}

        # new add
        self.logger = logger
        self.model = model
        self.value_model = value_model
        self.executor = HumanevalExecutor()
        self.use_embedding_diversity = use_embedding_diversity
        self.use_knowledge_base1 = use_knowledge_base1
        self.use_knowledge_base2 = use_knowledge_base2
        self.kb_coef = kb_coef
        self.use_ldb_debug = use_ldb_debug
        if self.use_embedding_diversity or self.use_knowledge_base1 or self.use_knowledge_base2:
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from langchain_huggingface import HuggingFaceEmbeddings
            # 加锁避免多线程同时加载模型
            with model_load_lock:
                self.embedding_model = HuggingFaceEmbeddings(
                    model_name="../../huggingface/bge-large-en-v1.5",
                    model_kwargs={'device': 'cuda'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                self.logger.info("嵌入模型已加载")
        if self.use_knowledge_base1:
            self.knowledge_base_1 = DocumentSearch(
                persist_directory="../../output/knowledge_base/kb_1_c_v2/chroma_db",
                embedding_model=self.embedding_model,
            )
        if self.use_knowledge_base2:
            self.knowledge_base_2 = DocumentSearch(
                persist_directory="../../output/knowledge_base/kb_2_v3/chroma_db",
                embedding_model=self.embedding_model,
            )

        # data
        self.question = data['prompt']
        self.given_tests = data['given_tests']
        self.entry_point = data['entry_point']
        self.question = self.question + '\npublic tests: \n' + '\n'.join(self.given_tests)  # 把given_tests拼接到question后面
        self.topic = self.get_problem_topic(self.question)

        # 埋点
        self.flags = {
            # 是否进入re_rollout
            're_rollout': False,
            # 是否定位错误步骤截断
            'locate_error_step': False,
            # 是否在过滤相似步骤后扩展两步
            'expand_multi_nodes': False,
        }

    def exec_code(self, code):
        exec_result = self.executor.execute_v2(code, self.given_tests, self.entry_point, timeout=5)
        self.logger.info(f"代码执行结果: reward={exec_result['reward']}")
        return exec_result['reward'], exec_result['results']

    def filter_similar_steps(self, sentences, similarity_threshold=0.85):
        if self.use_embedding_diversity:
            filtered = filter_similar_sentences(sentences, self.embedding_model, similarity_threshold)
            self.logger.info(f"步骤去重: 原始{len(sentences)}个步骤 -> 去重后剩{len(filtered)}个步骤")
            return filtered
        else:
            return sentences

    def get_problem_topic(self, question: str) -> str:
        prompt = get_topic_prompt_en.format(problem=question)
        response = self.model.generate(prompt)[0]
        result = extract_output_from_llm_response(response, tags_to_extract=['topic'])
        topic = result['topic']
        print(f'判断问题分类: {topic}')
        self.logger.info(f'判断问题分类: {topic}')
        return topic

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
        # 初始化返回值
        step_description = ""
        code_content = ""
        proposal = ""
        
        # 提取代码部分
        code_match = re.search(r'<code>(.*?)</code>', p, re.DOTALL)
        if code_match:
            code_content = code_match.group(1).strip()
        
        if self.lang == 'zh':
            if '下一步:' in p:
                stp = p.split('下一步:')[1].strip()
                # 检查步骤是否过短
                if len(stp) < 2:
                    print('输出步骤过短！\n')
                    return "", "", ""
                # 检查步骤是否重复
                if stp in y:
                    print('输出步骤重复！\n')
                    return "", "", ""
                
                # 处理数字编号格式
                pattern = r'\d\.\s*(.*)'
                match = re.search(pattern, stp)
                if match:
                    stp = match.group(1)
                
                step_description = '步骤' + str(step_n) + ': ' + stp
                
                # 构建完整的proposal
                if code_content:
                    proposal = step_description + '\n<code>\n' + code_content + '\n</code>\n'
                else:
                    proposal = step_description + '\n'
                
                print(f'标准化后新的步骤:{proposal}\n')
                return proposal, step_description, code_content

            elif '步骤' in p and ':' in p:
                pre_len = len(p.split(':')[0])
                p_ = p[pre_len:]
                p_ = p_.split('步骤')[0].strip()
                
                # 检查步骤是否过短
                if len(p_) < 3:
                    print('输出步骤过短！\n')
                    return "", "", ""
                # 检查步骤是否重复
                if p_[1:] in y:
                    print('输出步骤重复！\n')
                    return "", "", ""
                
                # 处理数字编号格式
                pattern = r'\d\.\s*(.*)'
                match = re.search(pattern, p_)
                if match:
                    p_ = match.group(1)
                
                step_description = '步骤' + str(step_n) + ': ' + p_
                
                # 构建完整的proposal
                if code_content:
                    proposal = step_description + '\n<code>\n' + code_content + '\n</code>\n'
                else:
                    proposal = step_description + '\n'
                
                print(f'标准化后新的步骤:{proposal}\n')
                return proposal, step_description, code_content
            else:
                print('输出格式有误！\n')
                return "", "", ""

        else:  # 英文处理
            # 提取Next step描述部分
            if "Next step" in p or 'Next Step' in p:
                match = re.search(r'Next [Ss]tep[:]\s*(.*?)(?=<code>|$)', p, re.DOTALL)
                if match:
                    step_description = match.group(1).strip()
            
            # 如果没有提取到Next step但有Step X:格式
            elif "Step" in p and ":" in p:
                match = re.search(r'Step \d+:\s*(.*?)(?=<code>|$)', p, re.DOTALL)
                if match:
                    step_description = match.group(1).strip()
                    # 移除可能的Step X:前缀
                    step_description = re.sub(r'Step \d+:\s*', '', step_description).strip()
            
            # 如果没有提取到描述但有Analysis:格式
            elif "Analysis:" in p:
                match = re.search(r'Analysis:\s*(.*?)(?=<code>|$)', p, re.DOTALL)
                if match:
                    step_description = match.group(1).strip()
            
            # 如果没有提取到任何描述内容
            if not step_description:
                return "", "", ""
            
            # 去除数字编号格式
            pattern = r'^\d\.\s*(.*)'
            match = re.search(pattern, step_description)
            if match:
                step_description = match.group(1)
            
            # 检查是否与现有步骤重复
            if step_description in y and not code_content:
                print('输出步骤重复！\n')
                return "", "", ""
            
            # 标准化步骤描述
            step_description = 'Step ' + str(step_n) + ': ' + step_description
            
            # 组合成标准化格式
            if code_content:
                proposal = step_description + '\n<code>\n' + code_content + '\n</code>\n'
            else:
                proposal = step_description + '\n'
            
            print(f'标准化后新的步骤:{proposal}\n')
            return proposal, step_description, code_content

    def get_full_step(self, y, step_n):
        prompt = self.get_full_step_prompt_wrap(self.question, y, step_n, self.lang)

        response = get_proposal(prompt, self)
        if not response:
            print('获得下一步失败！\n')
            return '', ''

        p = response
        prompt = self.extract_step_prompt_wrap(p, self.lang)
        response = get_proposal(prompt, self)

        # 依据<step></step>xml标签来提取内容
        full_steps_list = extract_steps_from_xml(response, step_n)

        prompt = generate_code_prompt_en.format(question=self.question, analysis=response)
        response_code = get_proposal(prompt, self)
        full_code = extract_python_code(response_code)
        return full_steps_list, full_code
    
    def re_get_full_step_with_reflection(self, wrong_code_with_reflection, existing_steps, step_n):
        
        prompt = rerollout_full_steps_with_reflection_prompt_en.format(problem=self.question, wrong_code_with_reflection=wrong_code_with_reflection, existing_steps=existing_steps)

        response = get_proposal(prompt, self)
        if not response:
            print('获得下一步失败！\n')
            return '', ''

        p = response
        prompt = self.extract_step_prompt_wrap(p, self.lang)
        response = get_proposal(prompt, self)

        # 依据<step></step>xml标签来提取内容
        full_steps_list = extract_steps_from_xml(response, step_n)

        prompt = generate_code_prompt_en.format(question=self.question, analysis=response)
        response_code = get_proposal(prompt, self)
        full_code = extract_python_code(response_code)
        return full_steps_list, full_code
    
    def value_full_step(self, solution, code, sandbox_result):
        # 获取solution'content'字段内容，并把所有内容拼在一起，使用换行区分
        solution = '\n'.join([s['content'] for s in solution])

        prompt = self.rollout_full_step_wrap(
            problem=self.question, 
            solution=solution,
            code=code,
            sandbox_result=sandbox_result,
            lang=self.lang,
        )
        response = get_value(prompt, self)
        
        # 依据<evaluation></evaluation>xml标签来提取得分
        pattern = r'<evaluation>(.*?)</evaluation>'
        match = re.search(pattern, response, re.DOTALL)
        try:
            score = float(match.group(1).strip())
        except Exception as e:
            print(f'提取<evaluation>标签失败')
            score = (self.low + self.high) / 2.0  # 默认分数为中间值
        
        # 返回response去除<evaluation></evaluation>xml标签的内容
        score_explain = re.sub(pattern, '', response, count=1).strip()
        return response, score_explain, score

    def remove_wrong_steps(self, steps_list, code, debug_ret, reflection):
        if debug_ret['is_success']:
            prompt = remove_wrong_steps_prompt_en.format(
                problem=self.question,
                solution='\n'.join([s['content'] for s in steps_list]),
                code=code,
                LDB_result=debug_ret['result'],
            )
        else:
            prompt = remove_wrong_steps_debug_failed_prompt_en.format(
                problem=self.question,
                solution='\n'.join([s['content'] for s in steps_list]),
                code=code,
                reflction=reflection
            )
            
        response = get_proposal(prompt, self)
            
        # 依据<step_n></step_n>xml标签来提取步骤数
        pattern = r'<step_n>(.*?)</step_n>'
        match = re.search(pattern, response, re.DOTALL)
        try:
            step_n = int(match.group(1).strip())
        except Exception as e:
            print('提取<step_n>标签失败')
            step_n = len(steps_list) - 1
            for i, step in enumerate(steps_list):
                if step['is_roll']:
                    step_n = i
                    break
        print(f'提取的步骤数: {step_n}')
        self.logger.info(f'提取的步骤数: {step_n}')
        self.flags['locate_error_step'] = True    # 埋点
        return steps_list[:step_n - 1], step_n
    
    # 对于模拟阶段生成失败的case，更新一次节点
    def renew_rollout_node(self, solution, wrong_code_with_reflection):
        prompt = renew_rollout_node_prompt_en.format(problem=self.question, solution=solution, wrong_code_with_reflection=wrong_code_with_reflection)
        response = get_proposal(prompt, self)
        
        # 依据<edit></edit>xml标签来提取内容
        pattern = r'<edit>(.*?)</edit>'
        match = re.search(pattern, response, re.DOTALL)
        try:
            # <edit>yes</edit>Step n:...
            if 'yes' in match.group(1).strip():
                is_edit = True
                # 提取去除<edit></edit>标签后的完整内容
                edit_content = response.replace(f'<edit>{match.group(1)}</edit>', '').strip()
            else:
                is_edit = False
                edit_content = ''
        except Exception as e:
            print('提取<edit>标签失败')
            is_edit = False
            edit_content = ''
        return is_edit, edit_content

    def get_next_step_use_reflection(self, y, step_n, reflection, history):  # 暂不支持 case-prompt
        # 加了历史信息，保证节点生成的diversity
        propose_prompt = self.zero_single_propose_wrap_use_reflection(
            self.question, y, step_n, reflection, self.lang, history
        )
        
        if self.use_knowledge_base1 and step_n == 1:
            filter = {"topic": self.topic} if self.topic else None
            search_results = self.knowledge_base_1.search_by_query_with_relevance_scores(self.question, k=1, score_threshold=0.5, filter=filter)
            print(f"检索知识库结果数量: {len(search_results)}")
            if search_results:
                content = "".join(
                    f"Problem:\n{r['problem']}\n"
                    f"Canonical steps:\n{r['canonical_steps']}\n" 
                    for r in search_results
                )
                content = "<content>\n" + content + "</content>\n"
                propose_prompt = self.zero_single_propose_wrap_use_knowledgebase(
                    self.question, self.lang, content, history
                )

        response = get_proposal(propose_prompt, self)
        if not response:
            print('获得下一步失败！\n')
            return '', '', ''

        print('-'*40 + 'Out get_next_step_use_reflection' + '-'*40)

        return self.extract_proposal(response, step_n, y)
      
    def get_step_value(self, y, step_n):
        if y in self.value_cache.keys():
            return self.value_cache[y]

        # llm分数
        prompt = self.value_prompt_wrap(self.question, y, self.lang)
        response = get_value(prompt, self)
        value = self.value_outputs_unwrap(response, self.lang, self.low, self.high)

        # 知识库分数
        if self.use_knowledge_base2:
            # 使用topic作为过滤条件
            filter = {"topic": self.topic} if self.topic else None
            
            # 使用steps_num作为过滤条件
            # step_n = min(step_n, 15)
            # filter = {"$and": [{"pre_steps_num": {"$gte": step_n - 1}}, {"pre_steps_num": {"$lte": step_n + 1}}]}
            
            search_results = self.knowledge_base_2.search_by_query_with_relevance_scores(self.question+'\n'+y, k=1, filter=filter)
            kb_score = search_results[0]['similarity'] if search_results else 0.0
            kb_score = kb_score * 10    # 映射到0-10范围
            value = (1 - self.kb_coef) * value + self.kb_coef * kb_score
            print(f"检索知识库2相似度得分: {kb_score}")

        self.value_cache.update({y: value})
        return value

    def run(self):
        self.clear_cache()
        self.set_limit_type()
        node, finish, root = MCTS(self)
        # vm
        if self.reward_model_type == 'vm':
            if self.sample_value != 'full':
                solution = node.y
                final_code = node.final_code
                summ = solution
                result = None
                final_answer = {'content': self.question, 'solution': solution, 'final_code': final_code, 'summary': summ, 'finish': finish,
                                'accurate': result, 'real_answer': self.answer}

                # MCTS运行完成
                print('-'*40 + 'MCTS finished!' + '-'*40)
                print(f"finish: {finish}")
                print(f"solution: {solution}")
                print(f"flags: {self.flags}")

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
                self.logger.info(f"MCTS运行完成: 叶节点数={len(leaf_nodes)}\n执行情况 = {json.dumps(self.flags, indent=4)}")
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

                # 获取node路径上所有
                y_list = []
                ptr_node = node
                while ptr_node != root:
                    y_list.append(ptr_node.pcd)
                    ptr_node = ptr_node.parent
                y_list.reverse()
                print("步骤长度: ", len(y_list))
                final_answer['y_list'] = y_list

                return final_answer, root