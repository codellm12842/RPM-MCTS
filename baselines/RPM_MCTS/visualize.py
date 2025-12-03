from datetime import datetime
from graphviz import Digraph

colors = ['Yellow', 'Gold', 'Orange', 'Orangered', 'Red', 'Crimson', 'Green']

def split_str(strs):
    sent_str = strs.split("。")
    all_strs = ''
    for sent in sent_str:
        piece_str = sent.split(",")
        for piece in piece_str:
            all_strs = all_strs + piece + '\n'
    return all_strs

def split_str_len(strs, length=100):
    all_strs = ''
    for i in range(0, len(strs), length):
        piece_str = strs[i:i + length]
        all_strs = all_strs + piece_str + '\n'
    return all_strs


def visualize(root, task, task_name, file_name="", file_suffix=""):
    # fname = f'graphs/{task_name}/{file_name}/{task.mode}/{task.propose_method}_{task.value_method}/{file_suffix}'
    cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fname = f"./graphs/{task_name}"
    g = Digraph("G", filename=fname, format='png', strict=False)
    str1 = 'Question: ' + split_str_len(task.question) \
        + '\nAccess sequence: ' + str(root.visit_sequence) \
        + '\nReflection: ' + split_str_len(str(root.reflection)) \
        + '\nVisits: ' + str(root.numVisits) \
        + '\nValue: ' + str(root.V) \
        + '\nflag: ' + str(root.final_ans_flag) \
        + '\nisFullyExpanded: ' + str(root.isFullyExpanded)
    
    g.node(str(root.visit_sequence), str1, color=colors[root.visit_sequence % len(colors)])
    sub_plot(g, root, task)
    g.node_attr['shape'] = 'tab'
    g.node_attr['fontname'] = 'Microsoft YaHei'
    g.graph_attr['size'] = '960,640'
    g.render(view=False)


def sub_plot(graph, root, task):
    if task.mode == 'mcts':
        for child in root.children.values():
            trans_str = split_str(child.pcd)
            str2 = split_str_len(trans_str) \
                + '\nAccess sequence: ' + str(child.visit_sequence) \
                + '\nReflection: ' + split_str_len(str(root.reflection)) \
                + '\nVisits: ' + str(child.numVisits) \
                + '\nValue: ' + str(child.V) \
                + '\nflag: ' + str(child.final_ans_flag) \
                + '\nisFullyExpanded: ' + str(child.isFullyExpanded)
                
            graph.node(str(child.visit_sequence), str2, color=colors[child.visit_sequence % len(colors)])
            graph.edge(str(root.visit_sequence), str(child.visit_sequence), str(child.visit_sequence - 1))
            sub_plot(graph, child, task)
    else:
        for child in root.children:
            trans_str = split_str(child.pcd)
            str2 = trans_str + '\nAccess sequence: ' + str(child.visit_sequence) + '\nValue: ' + str(child.V) + '\nflag: ' + str(
                child.final_ans_flag)
            graph.node(str(child.visit_sequence), str2, color=colors[child.visit_sequence % len(colors)])
            graph.edge(str(root.visit_sequence), str(child.visit_sequence), str(child.visit_sequence - 1))
            sub_plot(graph, child, task)
