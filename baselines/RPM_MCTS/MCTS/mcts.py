# 类型注解，支持mcts_task函数跳转同时避免循环导入
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from MCTS.task import MCTS_Task

import time
import math
import json
import random

from MCTS.base import treeNode
from prompt.prompts import generate_code_prompt_en
from rpm_mcts_tools.ldb_debug import ldb_debug
from rpm_mcts_tools.utils.utils import extract_python_code
random.seed(42)


def get_full_steps_roll(y, step_n, mcts_task: MCTS_Task):
    mcts_task.logger.info(f"开始获取完整步骤，从第{step_n}步开始")
    full_steps_list, full_code = mcts_task.get_full_step(y, step_n)
    mcts_task.logger.info(f"【完整步骤】\n{json.dumps(full_steps_list, indent=4)}")
    mcts_task.logger.info(f"【完整代码】\n{full_code}")
    return full_steps_list, full_code


def re_get_full_steps_roll(wrong_code_with_reflection, existing_steps, step_n, mcts_task: MCTS_Task):
    mcts_task.logger.info(f"基于反思重新生成完整步骤，从第{step_n}步开始")
    mcts_task.logger.info(f"\n反思内容:\n{wrong_code_with_reflection}")
    full_steps_list, full_code = mcts_task.re_get_full_step_with_reflection(wrong_code_with_reflection, existing_steps, step_n)
    mcts_task.logger.info(f"【基于反思完整步骤】\n{json.dumps(full_steps_list, indent=4)}")
    mcts_task.logger.info(f"【基于反思完整代码】\n{full_code}")
    return full_steps_list, full_code


def get_next_steps_expand(node: treeNode, mcts_task: MCTS_Task):
    mcts_task.logger.info(f"开始扩展节点，当前深度: {node.depth}")
    next_steps = []
    reflection = node.reflection
    history = []
    for i in range(mcts_task.branch):
        proposal, step_description, code_content = mcts_task.get_next_step_use_reflection(node.y, node.depth + 1, reflection, history)
        history.append(proposal)
        next_steps.append(proposal)
        mcts_task.logger.info(f"成功生成第{i+1}个扩展分支")
    next_steps = mcts_task.filter_similar_steps(next_steps)
    mcts_task.logger.info(f"扩展阶段生成{len(next_steps)}个有效分支：\n{json.dumps(next_steps, indent=4)}")
    # 埋点
    if len(next_steps) == 2:
        mcts_task.flags['expand_multi_nodes'] = True
    return next_steps


def update_node_and_parent_reflection(node: treeNode, update_reflection):
    # 更新当前节点和所有父节点的反思
    current_node = node
    updated_count = 0
    while current_node.depth >= 1:
        current_node.update_reflection(update_reflection)
        updated_count += 1
        current_node = current_node.parent


def add_roll_steps_to_tree(node: treeNode, mcts_task: MCTS_Task, steps_list, score, reflection):
    """将模拟步骤添加到MCTS树中并返回最后添加的节点。"""
    roll_steps = [action for action in steps_list if action["is_roll"]]
    if not roll_steps:
        mcts_task.logger.info("没有rollout步骤需要添加到树中")
        return node
    
    mcts_task.logger.info(f"开始将{len(roll_steps)}个rollout步骤添加到树中")
    current_node = node
    current_node.isFullyExpanded = True
    
    for i, action in enumerate(roll_steps):
        current_node.append_children(action["content"])
        child = current_node.children[action["content"]]
        child.numVisits = 1
        child.update_value(score)
        child.update_reflection(reflection)
        child.visit_sequence = mcts_task.node_count
        child.isFullyExpanded = True
        mcts_task.update_count()
        current_node = child

    # 叶节点的isFullyExpanded要设置为False
    current_node.isFullyExpanded = False
    mcts_task.logger.info(f"完成rollout步骤添加，最终叶节点深度: {current_node.depth}")
    return current_node


def fullstepsPolicy_without_exec_reward(node: treeNode, mcts_task: MCTS_Task):
    strs = node.y
    cur_step = node.depth + 1
    if node.reflection == '':
        full_steps_list, full_code = get_full_steps_roll(strs, cur_step, mcts_task)  # str_list
    else:
        full_steps_list, full_code = re_get_full_steps_roll(node.parent.reflection, node.y, node.depth + 1, mcts_task)  # str_list
    
    # 无测试用例
    response, score_explain, llm_value = mcts_task.value_full_step(full_steps_list, full_code, sandbox_result="(There are no test cases)")
    full_steps_score = llm_value

    if llm_value >= mcts_task.end_gate:
        current_node = add_roll_steps_to_tree(node, mcts_task, full_steps_list, full_steps_score, full_code + '\n' + score_explain)
        current_node.final_code = full_code
        return True, full_steps_score, current_node
    
    return False, full_steps_score, node


def fullstepsPolicy(node: treeNode, mcts_task: MCTS_Task):
    mcts_task.logger.info(f"执行完整步骤策略，节点深度: {node.depth}")
    strs = node.y
    cur_step = node.depth + 1
    if node.reflection == '':
        full_steps_list, full_code = get_full_steps_roll(strs, cur_step, mcts_task)  # str_list
    else:
        full_steps_list, full_code = re_get_full_steps_roll(node.parent.reflection, node.y, node.depth + 1, mcts_task)  # str_list
    
    mcts_task.logger.info("开始执行代码测试")
    exec_reward, sandbox_result = mcts_task.exec_code(full_code)
    mcts_task.logger.info(f"代码执行奖励: {exec_reward}")
    
    response, score_explain, llm_value = mcts_task.value_full_step(full_steps_list, full_code, sandbox_result)
    full_steps_score = 0.5 * exec_reward + 0.5 * llm_value
    mcts_task.logger.info(f"LLM评估分数: {llm_value}, 综合分数: {full_steps_score}")

    # 当通过公共测试用例，并且llm_value大于阈值的时候
    # 认为rollout阶段就可以做对，结束mcts
    if exec_reward == 10 and llm_value >= mcts_task.end_gate:
        mcts_task.logger.info(f"通过测试用例且达到LLM评估阈值，找到最终解决方案！")
        current_node = add_roll_steps_to_tree(node, mcts_task, full_steps_list, full_steps_score, full_code + '\n' + score_explain)
        current_node.final_code = full_code
        return True, full_steps_score, current_node
    
    # 通过公共测试用例，但llm_value小于阈值
    if exec_reward == 10:
        mcts_task.logger.info("通过测试用例但LLM评估分数不够，进行反思后重新rollout")
        # 反思继续rollout一次
        current_node = add_roll_steps_to_tree(node, mcts_task, full_steps_list, full_steps_score, full_code + '\n' + score_explain)
        current_node.final_code = full_code
        return re_rollout(current_node, mcts_task, full_code, score_explain)
    # 没通过公共测试用例
    else:
        mcts_task.logger.info("未通过测试用例，开始调试流程")
        # 调用ldb_debug进行调试
        if mcts_task.use_ldb_debug:
            mcts_task.logger.info("使用LDB调试器进行代码调试")
            debug_ret = {'is_success': False, 'result': ""}
            for res in sandbox_result:
                if res[0] == False:
                    cur_failed_test = res[1]['output']  # 获取第一个失败的测试用例
                    break
            try:
                debug_ret = ldb_debug(mcts_task.question, full_code, cur_failed_test, mcts_task.entry_point, mcts_task.model, messages="")
                mcts_task.logger.info("LDB调试成功")
            except Exception as e:
                mcts_task.logger.info(f"LDB调试失败: {e}")
        else:
            debug_ret = {'is_success': False, 'result': ""}

        # 移除错误的步骤，并得到是第几步出错step_n
        mcts_task.logger.info("开始移除错误步骤并生成反思")
        steps_list, step_n = mcts_task.remove_wrong_steps(full_steps_list, full_code, debug_ret, score_explain)
        if debug_ret['is_success']:
            rollout_reflection = debug_ret['result']
            mcts_task.logger.info("使用调试器生成的反思内容")
        else:
            rollout_reflection = score_explain
            mcts_task.logger.info("使用LLM评估生成的反思内容")
            
        current_node = add_roll_steps_to_tree(node, mcts_task, steps_list, full_steps_score, full_code + '\n' + score_explain)
        
        # 错误的步骤是已经扩展的叶节点的话，需更新节点，否则不需要更新
        if step_n == node.depth:
            mcts_task.logger.info(f"错误发生在当前节点深度{step_n}，尝试更新节点内容")
            is_edit, edit_content = mcts_task.renew_rollout_node(strs, full_code + '\n' + rollout_reflection)
            if is_edit:
                node.pcd = edit_content
                node.y = node.parent.y + node.pcd
                mcts_task.logger.info("成功更新节点内容")
            else:
                mcts_task.logger.info("节点内容无需更新")
                
        # 更新+反思继续rollout一次
        mcts_task.logger.info("基于反思重新进行rollout")
        return re_rollout(current_node, mcts_task, full_code, rollout_reflection)


# 得到反思后重新rollout
def re_rollout(node: treeNode, mcts_task: MCTS_Task, wrong_code, rollout_reflection):
    mcts_task.flags['re_rollout'] = True    # 埋点
    mcts_task.logger.info(f"开始重新rollout，节点深度: {node.depth}")
    mcts_task.logger.info(f"\n基于的反思内容:\n{rollout_reflection}")
    
    full_steps_list, full_code = re_get_full_steps_roll(wrong_code + '\n' + rollout_reflection, node.y, node.depth + 1, mcts_task)  # str_list
    exec_reward, sandbox_result = mcts_task.exec_code(full_code)
    mcts_task.logger.info(f"重新rollout后的执行奖励: {exec_reward}")
    
    response, score_explain, llm_value = mcts_task.value_full_step(full_steps_list, full_code, sandbox_result)
    full_steps_score = 0.5 * exec_reward + 0.5 * llm_value
    mcts_task.logger.info(f"重新rollout后的综合分数: {full_steps_score}")
    
    # 当通过公共测试用例，并且llm_value大于阈值的时候，结束mcts
    if exec_reward == 10 and llm_value >= mcts_task.end_gate:
        mcts_task.logger.info("重新rollout成功找到解决方案！")
        current_node = add_roll_steps_to_tree(node, mcts_task, full_steps_list, full_steps_score, full_code + '\n' + score_explain)
        current_node.final_code = full_code
        return True, full_steps_score, current_node
    # 通过公共测试用例，但llm_value小于10
    if exec_reward == 10:
        mcts_task.logger.info("重新rollout通过测试但LLM分数不够，更新反思信息")
        update_node_and_parent_reflection(node, full_code + '\n' + score_explain)
        current_node = add_roll_steps_to_tree(node, mcts_task, full_steps_list, full_steps_score, full_code + '\n' + score_explain)
        current_node.final_code = full_code
        return False, full_steps_score, None
    # 没通过公共测试用例
    elif exec_reward < 10:
        mcts_task.logger.info("重新rollout仍未通过测试，更新反思信息")
        update_node_and_parent_reflection(node, full_code + '\n' + score_explain)
        return False, full_steps_score, None


def rollPolicy(node: treeNode, mcts_task: MCTS_Task):
    mcts_task.logger.info(f"执行rollout策略: {mcts_task.roll_policy}")
    if mcts_task.roll_policy == 'greedy':
        return greedyPolicy(node, mcts_task)
    elif mcts_task.roll_policy == 'random':
        return randomPolicy(node, mcts_task)
    elif mcts_task.roll_policy == 'fullsteps':
        return fullstepsPolicy(node, mcts_task)
    elif mcts_task.roll_policy == 'fullsteps_without_exec_reward':
        return fullstepsPolicy_without_exec_reward(node, mcts_task)
    else:
        raise ValueError(f"Unknown policy: {mcts_task.roll_policy}")


def MCTS_search(mcts_task: MCTS_Task):
    root = treeNode('')

    if mcts_task.limit_type == 'time':
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        round_count = 0
        while time.time() < timeLimit:
            round_count += 1
            elapsed_time = time.time() - time_start
            print(f'<开始新搜索轮次，目前总时间:{elapsed_time}>\n')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                return root, node, elapsed_time
    else:
        for i in range(mcts_task.iteration_limit):
            mcts_task.logger.info(f'<<<开始新搜索轮次，目前已完成轮次数:{i}>>>')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                mcts_task.logger.info(f"在第{i+1}轮找到解决方案！")
                return root, node, i + 1
        mcts_task.logger.info(f"完成{mcts_task.iteration_limit}轮搜索，未找到满足条件的解决方案")
    return root, None, None


def executeRound(root: treeNode, mcts_task: MCTS_Task):
    print('-' * 40)
    print('选择阶段\n')
    mcts_task.logger.info("--- 选择阶段 ---")
    node = selectNode(root, mcts_task)

    print('-' * 40)
    print('扩展阶段\n')
    mcts_task.logger.info("--- 扩展阶段 ---")
    node = expand(node, mcts_task)

    print('-' * 40)
    print('模拟搜索阶段\n')
    mcts_task.logger.info("--- 模拟阶段 ---")
    end_flag, end_node, roll_node = simulate(node, mcts_task)
    if end_flag:
        return True, end_node, root

    print('-' * 40)
    print('反向传播阶段\n')
    mcts_task.logger.info("--- 反向传播阶段 ---")
    back_propagate(node)
    return False, node, root


def selectNode(node: treeNode, mcts_task: MCTS_Task):
    selection_path = []
    while node.isFullyExpanded:
        node = getBestChild(node, mcts_task)
        selection_path.append(f"节点{node.visit_sequence}(深度{node.depth})")
        mcts_task.logger.info(f'选中节点：{node.visit_sequence}，深度：{node.depth}，访问次数：{node.numVisits}，价值：{node.V:.3f}')
        print(f'选中节点：{node.visit_sequence}，深度：{node.depth}，访问次数：{node.numVisits}，价值：{node.V}\n')
    mcts_task.logger.info(f"选择路径: {' -> '.join(selection_path)} -> 选定节点{node.visit_sequence}")
    return node


def expand(node: treeNode, mcts_task: MCTS_Task):
    actions = get_next_steps_expand(node, mcts_task)

    added_children = 0
    for action in actions:
        if action not in node.children.keys():
            node.append_children(action)
            child = node.children[action]
            value = mcts_task.get_step_value(child.y, child.depth)
            child.update_value(value)
            child.visit_sequence = mcts_task.node_count
            mcts_task.update_count()
            added_children += 1
            mcts_task.logger.info(f"添加子节点{child.visit_sequence}，初始价值: {value:.3f}")
    
    node.isFullyExpanded = True
    mcts_task.logger.info(f"节点扩展完成，新增{added_children}个子节点")
    return node


def simulate(node: treeNode, mcts_task: MCTS_Task):
    roll_node = getBestChild(node, mcts_task)
    mcts_task.logger.info(f"选择最优子节点进行rollout")
    end_flag, best_V, end_node = rollPolicy(roll_node, mcts_task)
    if end_flag:
        mcts_task.logger.info(f"rollout阶段找到解决方案！")
        print(f'找到解决方案，返回叶节点：深度={end_node.depth}')
        return True, end_node, roll_node
    # 更新当前节点的价值
    old_value = roll_node.V
    roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
    roll_node.numVisits += 1
    mcts_task.logger.info(f"更新rollout节点价值: {old_value:.3f} -> {roll_node.V:.3f}")
    return False, None, roll_node


def back_propagate(node: treeNode):  # 反向传播
    propagation_path = []
    original_node = node
    while node is not None:
        old_value = node.V
        old_visits = node.numVisits
        # 更新访问次数
        node.numVisits += 1
        if node.isFullyExpanded:
            child_Vs = [child.V * child.numVisits for child in node.children.values()]
            total_num_visits = sum([child.numVisits for child in node.children.values()])
            if total_num_visits > 0:
                # 使用加权平均来更新当前节点的价值
                node.V = sum(child_Vs) / total_num_visits
        propagation_path.append(f"节点{getattr(node, 'visit_sequence', 'root')}(价值{old_value:.3f}->{node.V:.3f})")
        node = node.parent


def getBestChild(node: treeNode, mcts_task: MCTS_Task):
    mcts_task.logger.info(f"为节点{getattr(node, 'visit_sequence', 'root')}选择最佳子节点")
    bestValue = mcts_task.low
    bestNodes = []
    child_info = []
    for child in node.children.values():
        # 计算UCB值，此处INF为10
        nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
            2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
        child_info.append(f"节点{child.visit_sequence}(UCB:{nodeValue:.3f})")
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    
    selected = random.choice(bestNodes)
    mcts_task.logger.info(f"子节点UCB值: {', '.join(child_info)}")
    mcts_task.logger.info(f"选择最佳子节点{selected.visit_sequence}，UCB值: {bestValue:.3f}")
    return selected


def MCTS(mcts_task: MCTS_Task):
    root, node, finish = MCTS_search(mcts_task)

    if mcts_task.sample_value == 'full':
        mcts_task.logger.info('MCTS采样完成')
        print('采样完成。\n')
        return None, -1, root
    else:
        if mcts_task.reward_model_type == 'vm':
            if finish is not None:
                mcts_task.logger.info(f'找到最终解决方案！节点深度: {node.depth}')
                print(f'已找到最终解!\nSolution:{node.y}\n')
                return node, finish, root
            else:
                mcts_task.logger.info("在规定时间/轮次内未找到满足要求的解答，寻找最高价值解答")
                # dfs遍历树的所有节点，找到有final_code属性的节点且价值最高
                def find_best_final_code_node(node):
                    best_node = None
                    best_value = -float('inf')
                    
                    # 检查当前节点是否有final_code属性
                    if hasattr(node, 'final_code') and node.final_code is not None:
                        best_node = node
                        best_value = node.V
                    
                    # 递归检查所有子节点
                    for child in node.children.values():
                        child_best_node, child_best_value = find_best_final_code_node(child)
                        if child_best_node is not None and child_best_value > best_value:
                            best_node = child_best_node
                            best_value = child_best_value
                    
                    return best_node, best_value
                best_node, best_V = find_best_final_code_node(root)
                
                # 如果没有找到final_code属性的节点，则使用getBestV方法获取最佳节点
                if best_node is None:
                    mcts_task.logger.info("未找到包含final_code的节点，使用getBestV方法")
                    best_node, best_V = root.getBestV()
                else:
                    mcts_task.logger.info(f"找到最佳final_code节点，价值: {best_V:.3f}")
                
                mcts_task.logger.info(f"\n采用最高价值解答:\n{best_node.y}")
                print(f'在规定时间/轮次内未找到满足要求价值的解答，采用最高价值价值解答代替。\nSolution:{best_node.y}\n')

                # generate code
                mcts_task.logger.info("为最佳节点生成最终代码")
                prompt = generate_code_prompt_en.format(question=mcts_task.question, analysis=best_node.y)
                response = mcts_task.model.generate(prompt, temperature=0.7)[0]
                code = extract_python_code(response)
                best_node.final_code = code
                mcts_task.logger.info(f"\n生成的最终代码:\n{code}")
                return best_node, -1, root
        else:
            mcts_task.logger.info('当前不支持的奖励模型类型，采样结束')
            print('尚未支持解答选择，采样结束。\n')
            return None, -1, root
