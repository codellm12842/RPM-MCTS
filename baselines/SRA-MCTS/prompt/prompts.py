CODE_analyze_summary_prompt = '''
Given a code analyze problem and its corresponding solution, your task is to extract the answer obtained in the solution.
You should summarize the answer using the format: "The final answer is ...". Replace "..." with the answer obtained in the solution.
Problem: '''

CODE_analyze_summary_prompt_zh = '''

'''

cot_prompt = '''
You are supposed to provide a solution to a given problem.
Both natural language analysis and code are needed.
Let's think step by step.
Problem:
{question}
Solution: 
'''

completed_prompt_en = '''
You will receive a problem along with its corresponding solution steps. 
We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
All the substeps should be combined in a way that avoids contradictions, forming a coherent solution to the original complex problem.
Please evaluate whether these steps effectively solve the problem based on the following criteria:

	1.	Correctness: Are the steps factually accurate and aligned with the problem requirements?
	2.	Completeness: Do the steps cover all necessary aspects needed to fully address the problem?

Both criteria must be satisfied for the problem to be considered solved.

	•	If the solution steps fail to meet either criterion, please respond with: <unsolved>.
	•	If the solution steps adequately resolve the problem, please respond with: <solved>.
Problem:
{problem}
Solution:
{solution}
@@@ Your response:
'''

completed_prompt_zh = '''
你将收到一个问题和对应的解决步骤，使用你的知识判断所给的解决步骤能否解决这个问题。
如果解决步骤能够解决这个问题，请回答'可以解决'；如果不能解决，请回答'不能解决'。
'''

code_proposal_prompt_zh = '''
你的任务是给定一个编程问题和已有的解答步骤（并不是完整的答案），给出正确的下一步。
你分析的步骤将作为输入提供给代码大模型进行代码生成，请按照代码大模型便于理解的方式生成你的推理步骤，不要在一步中包括太多的内容，做到条理清晰，分步合理。
注意: 禁止你写出代码，只用自然语言分析推理步骤。如果你写代码，你的回答将会被丢弃。

假设输入为n步，则输入的格式为:
"题目:...
已有步骤:
步骤1:...
步骤2:...
...
步骤n:..."

其中...表示省略的输入信息。
如果n等于0，你需要从头开始简单地分析解题思路，然后输出第一步。如果n不等于0，那么你需要对输入部分的解题方法进行简短的分析，然后依照已有步骤的思路和分析，输出你认为正确的下一步骤(第n+1步)。
输出格式限定为:

"分析:...
下一步:..."

其中...表示省略的输出信息，这是你应该填充的部分。
下面是输入，请你按照限定的输出格式进行输出，不要输出多余的信息，不要复述题目。

题目:'''

code_proposal_prompt_use_reflection_zh = '''
你的任务是根据给定的编程问题和已有的解答步骤（并不是完整的答案），推理并生成当前问题的下一步操作。
•	每次生成的步骤必须简短且明确，专注于解决方案的一小部分，不要过于复杂。
•	整个解决方案应至少包含三步，因此不要跳过任何必要的步骤。
•	你的输出应条理清晰，每次只描述一个推理步骤，避免在单一步骤中包含多个推理点。
•	注意： 你只能用自然语言描述推理步骤，不能输出代码。如果你输出代码，回答将被丢弃。

假设输入为n步，则输入的格式为:
"题目:...
已有步骤:
步骤1:...
步骤2:...
...
步骤n:...
意见:...
"
其中...表示省略的输入信息。
如果n等于0，你需要按照意见部分给定的解题思路给出解答这道题的第一步。如果n不等于0，你需要按照已有步骤的思路和意见部分给出的提示，输出完整且清晰的下一个解答步骤(第n+1步)，不允许输出后面的步骤(第n+2步)。如果意见部分为空，就按照已有步骤的思路直接输出下一步(第n+1步)。
输出格式限定为:
"下一步:..."
其中...表示省略的输出信息，这是你应该填充的下一步解答。
下面是输入，请你按照规定格式进行输出。

题目:'''

code_proposal_prompt_en = '''
Your task is to infer and generate the next step of a programming solution based on the given problem and the partial steps already provided (these are not the complete answer).
Let's think step by step. But you only generate one step at a time.
We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
All the substeps should be combined in a way that avoids contradictions, forming a coherent solution to the original complex problem.

**Input format (n steps):**
Problem:
Existing steps:
Step 1:
Step 2:
...
Step n:
Where "..." denotes omitted input information.

If n is equal to 0, you need to start from scratch and analyze the solution idea briefly, and then output the first step. 
Otherwise, you need to output the correct next step of the existing solution, following the ideas of the existing steps. 
Each step generated should be concise and focused, addressing only a small part of the solution. Avoid making the steps too complex or combining multiple ideas into one.
The complete solution should consist of at least three steps, so don't skip any essential steps.
Your output should be clear and systematic, with each step described one at a time to ensure logical progression.
Note: You are only allowed to describe the reasoning steps in natural language. Do not output any code. 
**If your answer includes code, it will cause unforeseen losses!**

**Output format:**
Next step: ...

where ... indicate the omitted output information, which is the part you should fill in.
The following is the input. Please output according to the specified output format, do not output unnecessary information, and do not repeat the question.
Problem:'''

code_proposal_prompt_use_reflection_en = '''
Your task is to provide the correct **next step** based on the given analysis for a given programming problem and its existing solution steps (which are incomplete).  
Let's think step by step. But you only generate one step at a time.
We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
All the substeps should be combined in a way that avoids contradictions, forming a coherent solution to the original complex problem.

**Input format (n steps):**

Problem:
Existing steps:
Step 1:
Step 2:
...
Step n:
Analysis: ...

Where "..." denotes omitted input information.

- The steps you generate will be passed to a code generation model, so they should be structured in a way that is easy for the model to understand.  
- Keep each step concise and focused, avoiding the inclusion of too much information at once. Ensure clear organization and logical progression in your reasoning.  
- **Important:** You can use very little code as detailed explanations in your answers, but you cannot just write code.
- **If your answer includes code, it will cause unforeseen losses!**
- Your answer should be based on the given analysis. Only if the analysis is wrong can you answer it in your own way.

- If no existing steps are provided, you should output the first step based on the given analysis.  
- If there are existing steps, output the next step (Step n+1) that logically follows the provided analysis and the previous steps.

**Output format:**
Next step: ...

Where "..." is the next reasoning step you should fill in. This should be a clear and complete reasoning step, possibly including calculations, analysis, or decision-making.

**Here is the input. Please follow the restricted output format.**
Problem:
'''

single_reflection_prompt_en = '''
You are an expert. Given a programming problem that has been partially solved, your task is to evaluate the provided steps and offer guidance. 
We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
All the substeps should be combined in a way that avoids contradictions, forming a coherent solution to the original complex problem. 

- First, determine whether the given steps have fully addressed the problem.  
- If the problem is not completely solved, provide a brief evaluation of the existing steps based on your expertise and suggest a concise direction for the next step.

You need to handle two scenarios and provide the corresponding output:
1. **If the problem has been fully solved:** Output only "<end>". No additional information is necessary.
2. **If the problem is not fully solved:** Offer an analysis of the existing steps and suggest a brief direction for the next step. If no steps are provided, give a direction for the first step instead.

**Output format:**

Analysis: ...

Where "..." represents your evaluation and suggestion.

**Here is the input. Please follow the required output format and do not attempt to solve the entire problem.**

Problem:
'''

single_reflection_prompt_zh = '''
你是一个专家，给定一个编程问题，我已经完成了部分解答，需要你给一些提示。你需要先判断给定的步骤是否已经解决问题，如果还没有解决，请你基于你的已有知识储备给出针对已有步骤的简单评价和下一步的简要思路。

你需要区分两种情况给出对应的输出:
1.如果给定的步骤已经解决了题目并且给出了答案，那么请直接输出:"<end>"即可，不需要输出其他内容。
2.如果还没有完全解决题目，你需要针对已有步骤给出意见，然后基于你的已有知识给出下一步的简要思路。如果输入没有提供任何已有步骤，那么你只需给出第一步的简要思路。
输出格式限定为:
"意见:..."，其中...表示省略的输出信息，这是你应该填充的部分。
下面是输入，请你按照要求的输出方式进行输出，不要试图解答整个题目。

题目: '''

single_reflection_prompt_simple = '''
你是一个专家，给定一个编程题目和一些相应的解答步骤（不一定完整）。你需要判断给定的步骤是否已经解决问题并给出答案。

你需要区分两种情况给出对应的输出:
1.如果给定的步骤已经分析出了题目要求的所需步骤，那么请直接输出:"<end>"，不需要输出其他内容。
2.如果给定步骤还没有分析出所需步骤，那么请直接输出:"<continue>"即可，不需要输出其他内容。
注意，如果现有步骤没有按题目要求分析出答案，那么应当视为未解决。
下面是输入，请你按照要求的输出方式进行输出，你不需要解答题目。

题目: '''

single_reflection_prompt_simple_en = '''
Please assess whether the given steps have resolved the problem as follows:

Read and analyze the following problem: ```{problem}```.

Read and analyze the provided solution steps: ```{steps}```.

If the given steps have not adequately analyzed or resolved the necessary steps for the problem, please output: "<continue>". No additional content is needed.
If the given steps have adequately analyzed and resolved the necessary steps for the problem, please output: "<end>". No additional content is needed.
'''

# 对已有步骤进行打分（增加xml标签）
critic_en = '''
Your role is to act as an evaluator, and your task is to assess whether the proposed solution effectively addresses the problem.
We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
All the substeps should be combined in a way that avoids contradictions, forming a coherent solution to the original complex problem.

Do not attempt to answer this question or write code, only output the score

If the solution can successfully resolve the issue, give it a score of 10. 
If it cannot solve the problem yet but does not contain any incorrect steps, and adding a few new steps can resolve the issue, give it a score between 5 and 7. 
If the final step of the solution includes an error, the score should be below 3. 
If there is an error in a step that is not the final one, the score should be between 3 and 5.

The more mistakes in all steps, the closer the score should be to 0 . The closer all steps are to a correct solution, the closer the score should be to 10 . 

A score of 6 or higher should only be given if all previous steps are correct. 
A score of 10 should be given only if all previous steps are entirely correct and effectively solve the problem.
First, generate an analysis, and then give a score. Your analysis and scoring should be based entirely on the given steps without generating further steps. 
Please study the following example's format.

**Example 1**
Problem: You are given a string `s` of length `n` where `s[i]` is either: *   `'D'` means decreasing, or *   `'I'` means increasing. A permutation `perm` of `n + 1` integers of all the integers in the range `[0, n]` is called a **valid permutation** if for all valid `i`: *   If `s[i] == 'D'`, then `perm[i] > perm[i + 1]`, and *   If `s[i] == 'I'`, then `perm[i] < perm[i + 1]`. Return _the number of **valid permutations**_ `perm`. Since the answer may be large, return it **modulo** `109 + 7`.
Existing Steps:
Step 1: We can approach the problem using dynamic programming. We maintain a state dp[i] that represents the number of valid ways to form permutations up to the i-th position.
Step 2: The decision at each step is influenced by whether the character at s[i] is 'D' or 'I'. For 'D', we need to choose a value smaller than the current one, and for 'I', we need to choose a value larger.
Step 3: We will iterate over the positions and update the possible permutations dynamically based on the previous states.
Step 4: Given that the number of permutations can grow large, we ensure that the final result is taken modulo 10^9 + 7 to avoid overflow.
Score:
<score>10</score>

**Example 2**
Problem: You are given a string s consisting of 'a' and 'b' characters. Your task is to count how many distinct substrings can be formed from s. The answer can be large, so return the result modulo 10^9 + 7.

Existing Steps:
Step 1: Substrings identification: A substring is a continuous sequence of characters within the string. We need to identify all possible substrings of the given string s.
Step 2: Count distinct substrings: From all identified substrings, we need to count how many of them are unique.
Score:
<score>6</score>

Below is a given problem and the existing steps. Provide a score based on the principles. 
Note not to generate further steps in the analysis, the score should be based entirely on the given steps.
Wapper the score with the <score></score> tag. And only output one score.

Problem:
{problem}
Existing Steps:
{existing_steps}
Score:
'''

# critic_en = '''
# Your role is to act as an evaluator, and your task is to assess whether the proposed solution effectively addresses the problem.
# We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
# All the substeps should be combined in a way that avoids contradictions, forming a coherent solution to the original complex problem.

# Do not attempt to answer this question or write code, only output the score

# If the solution can successfully resolve the issue, give it a score of 10. 
# If it cannot solve the problem yet but does not contain any incorrect steps, and adding a few new steps can resolve the issue, give it a score between 5 and 7. 
# If the final step of the solution includes an error, the score should be below 3. 
# If there is an error in a step that is not the final one, the score should be between 3 and 5.

# The more mistakes in all steps, the closer the score should be to 0 . The closer all steps are to a correct solution, the closer the score should be to 10 . 

# But if the complete code appears in the solution, give 0 points.
# A score of 6 or higher should only be given if all previous steps are correct. 
# A score of 10 should be given only if all previous steps are entirely correct and effectively solve the problem.
# First, generate an analysis, and then give a score. Your analysis and scoring should be based entirely on the given steps without generating further steps. 
# Please study the following example's format.

# **Example 1**
# Problem: You are given a string `s` of length `n` where `s[i]` is either: *   `'D'` means decreasing, or *   `'I'` means increasing. A permutation `perm` of `n + 1` integers of all the integers in the range `[0, n]` is called a **valid permutation** if for all valid `i`: *   If `s[i] == 'D'`, then `perm[i] > perm[i + 1]`, and *   If `s[i] == 'I'`, then `perm[i] < perm[i + 1]`. Return _the number of **valid permutations**_ `perm`. Since the answer may be large, return it **modulo** `109 + 7`.
# Existing Steps:
# Step 1: We can approach the problem using dynamic programming. We maintain a state dp[i] that represents the number of valid ways to form permutations up to the i-th position.
# Step 2: The decision at each step is influenced by whether the character at s[i] is 'D' or 'I'. For 'D', we need to choose a value smaller than the current one, and for 'I', we need to choose a value larger.
# Step 3: We will iterate over the positions and update the possible permutations dynamically based on the previous states.
# Step 4: Given that the number of permutations can grow large, we ensure that the final result is taken modulo 10^9 + 7 to avoid overflow.

# Score: 10

# **Example 2**
# Problem: You are given a string s consisting of 'a' and 'b' characters. Your task is to count how many distinct substrings can be formed from s. The answer can be large, so return the result modulo 10^9 + 7.

# Existing Steps:
# Step 1: Substrings identification: A substring is a continuous sequence of characters within the string. We need to identify all possible substrings of the given string s.
# Step 2: Count distinct substrings: From all identified substrings, we need to count how many of them are unique.

# Score: 6

# Below is a given problem and the existing steps. Provide a score based on the principles. 
# Note not to generate further steps in the analysis, the score should be based entirely on the given steps.
# The output format should be limited to: 
# "Score:..."
# where ... indicates the omitted output content that you need to fill in.

# Problem:
# {problem}
# Existing Steps:
# {existing_steps}
# Score: '''

critic_zh = '''
你的任务是根据给定的编程问题和已有的解答步骤，判断其中的最后一步是否能够顺利解决该问题，并对所给的整个解决方案进行打分。
打分应该是0到10之间的整数。
如果所给的解答方案已经能正确地解决这个问题，给10分;
如果最后一步与前面的步骤逻辑通顺，并且按照这个步骤继续推理能顺利解决问题，给分应大于等于7, 但小于9;
如果最后一步是解题的未来步骤之一，但与前面的步骤不连续，给5分;如果最后一步与前面步骤连续，但不能解决问题，给3分;
如果既不连续，又不能解决问题，给0分。
所有的步骤错的越多，分数越接近 0 分。所有的步骤越接近正确分析，分数越接近 10 分。
给大于或等于9分必须是已有步骤全部正确，并且加上最后一步后能顺利解决这个问题。
先生成分析，后给出分数，你的分析和给分应该全部基于输入给定的步骤，不要继续生成下面的步骤。请学习以下样例。

输入:
"问题: 找到一个整数的最大质因数
已有步骤: 
步骤1:了解质数的概念。质数是大于1的数，除了1和自身之外没有其他除数。
步骤2:我们需要定义一个函数来检查一个数字是否为质数。这个函数将接受一个整数作为输入，如果数字是质数，则返回True，否则返回False。
步骤3:我们需要定义一个函数来找到x的最大质因数。这个函数将接受一个整数作为输入，并返回该整数的最大质因数。"

输出:
"分析: 步骤3为最后一步，它的内容与题目部分相关，但与前面的步骤不连续，并且不能解决问题，得分不能超过1。正确的步骤应该是接着上一步对函数功能和接口的定义，给出具体的操作方法。
分数: 1"

输入:
"问题: 给定一个整数数组，找出和为指定值的两个数。
已有步骤:
步骤1: 了解两数之和问题的概念。两数之和问题要求在一个数组中找到两个数，使它们的和等于指定值。
步骤2: 定义一个函数来解决这个问题。该函数将接受一个整数数组和一个目标值作为输入，并返回两个数的索引。
步骤3: 使用双重循环遍历数组中的每一对数，并检查它们的和是否等于目标值。"

输出:
"分析: 步骤3为最后一步，它的内容与题目相关，并且与前面的步骤逻辑连续。但是，使用双重循环遍历数组中的每一对数能够解决问题，但效率较低，不能被认为是最优解，因此步骤3不能得到满分，但应高于7。
分数: 8"

输入:
"问题: 反转一个单链表。
已有步骤:
步骤1: 了解单链表的结构。单链表是由一系列节点组成的线性数据结构，每个节点包含数据和指向下一个节点的指针。
步骤2: 定义一个函数来反转单链表。该函数将接受一个单链表的头节点作为输入，并返回反转后的链表的头节点。
步骤3: 使用递归方法来反转链表。"

输出:
"分析: 步骤3为最后一步，它的内容与题目相关，并且与前面的步骤逻辑连续。但是，使用递归方法反转链表描述的过于简洁，没有细节地说明实现的具体操作方法。所以分数应大于5，但小于7。
分数: 6"

输入:
"问题: 给定一个二叉树，找到其最大深度。
已有步骤:
步骤1: 了解二叉树的基本概念。二叉树是每个节点最多有两个子节点的树结构。
步骤2: 定义一个函数来计算二叉树的最大深度。该函数将接受一个二叉树的根节点作为输入，并返回树的最大深度。
步骤3: 使用递归遍历树的每个节点，并计算每个子树的深度。"

"分析: 步骤3为最后一步，它的内容与题目相关，并且与前面的步骤逻辑连续。最后一步加上前面的步骤能够完全地解决这个问题，故给1分。
分数: 10"

输入:
"问题: 判断一个字符串是否是回文。
已有步骤:
步骤1: 了解回文的概念。回文是指正着读和反着读都相同的字符串。
步骤2: 遍历字符串并比较首尾字符，如果相同则继续，否则返回False。"

输出:
"分析: 步骤3为最后一步，它的内容与题目相关并且操作正确，但与前面的步骤不连续。前面的步骤介绍了回文的定义，但具体分析题目的要求。应该补充步骤:定义一个函数，输入一个字符串，当输入为回文串时返回True，否则返回False。尽管步骤3能解决问题，但由于与前面的步骤不连续，得分不能达到最高。
分数: 3"

下面给定一个问题和已有的步骤，给出分析和打分。注意不要在分析中输出接下来的步骤，评分应该完全依据输入给定的步骤。
输出格式限定为:
"分析:...
分数:..."
其中...表示省略的输出内容，这是你需要填充的部分。

输入:
问题: '''

self_critic_prompt = '''
Given a problem and an existing solution, your task is to evaluate the correctness of the solution and provide an evaluation score. 
Your output should be a integer ranging from 0 to 10. The more correct the solution is, the higher your evaluation score should be.

Problem:'''
