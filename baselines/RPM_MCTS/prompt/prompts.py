get_topic_prompt_en = '''problem:
{problem}

Your task is to select the most relevant topic from the following topics based on the programming problem.
Note: Analyze first, then output the most relevant topic wrapped in <topic></topic> tags such as <topic>Data Structures</topic>.

topics:
- Data Structures
- Algorithm Strategies
- String Processing
- Sorting and Searching
- Graph Theory
- Bit Manipulation
- Mathematics and Number Theory
- Computational Geometry
- Optimization Problems
- Two-Pointer Techniques
- Dynamic Programming
- Recursion and Backtracking
- Hashing Techniques
- Other
'''

# 模拟阶段生成完整答案
code_proposal_prompt_en_v2 = '''
Your task is to take a programming problem and incomplete solution steps (not a full answer), then continue from the provided steps to complete all remaining steps and generate the complete final solution.

Let's think step by step. We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
All the substeps should be combined in a way that avoids contradictions, forming a coherent solution to the original complex problem.
Note: Do not modify the existing solution steps.

**Input format (n steps):**
Problem:
Existing steps:
Step 1:
Step 2:
...
Step n:
Where "..." denotes omitted input information.

If n is equal to 0, you need to start from scratch and analyze the solution idea briefly, and then output the complete answer. 
Otherwise, you need to output the complete answer that you think is correct following the train of thought of the existing steps.
Each step generated should be concise and focused, addressing only a small part of the solution. Avoid making the steps too complex or combining multiple ideas into one.
The complete solution should consist of at least three steps, so don't skip any essential steps.
Your output should be clear and systematic, with each step described one at a time to ensure logical progression.
Note: You are only allowed to describe the reasoning steps in natural language. Do not output any code. 
**If your answer includes code, it will cause unforeseen losses!**

**Output format:**
Step 1:
Step 2:
...
Step n:
Step n + 1:
Step n + 2:
...
Among them, Step 1 to Step n are consistent with the existing steps. Continue to generate based on the existing steps to obtain a complete answer.
The following is the input. Please output according to the specified output format, do not output unnecessary information, and do not repeat the question.
Note: Your output should start from Step 1 and include all the steps, not just the next step.

Problem:
'''

renew_rollout_node_prompt_en = '''Python code problem:
{problem}
Thoughts:
{solution}
Wrong code and reflection:
{wrong_code_with_reflection}

Your task is to take a programming problem and incomplete solution steps (not a full answer), and determine whether the last thought step in the provided solution needs to be modified based on the previous incorrect code used to solve the problem and a reflection given to you. If it needs to be modified, output <edit>yes</edit> enclosed in XML tags, and provide the modification content. If it doesn't need to be modified, output <edit>no</edit>.
- Note: Only modify the last thinking step. Do not modify the other thinking steps. And only output the thinking content of the last step.
- Only "yes" or "no" are allowed to be output within the <edit></edit> tag. Do not output any other content.

**Output format:**
<edit>yes</edit>Step n:...
<edit>no</edit>
Among them,... represents the thinking content of the modified last step.

Output:
'''

rerollout_full_steps_with_reflection_prompt_en = '''Your task is to handle a programming problem. You will be given a wrong code to solving this problem previously and a reflection. Please think again and address the problems that existed before. Your goal is to continue from the provided steps to complete all remaining steps and generate the complete final solution.

Let's think step by step. We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
All the substeps should be combined in a way that avoids contradictions, forming a coherent solution to the original complex problem.
Note: Do not modify the existing solution steps.

**Input format (n steps):**
Problem:
Wrong code and reflections:
Existing steps:
Step 1:
Step 2:
...
Step n:
Where "..." denotes omitted input information.

If n is equal to 0, you need to start from scratch and analyze the solution idea briefly, and then output the complete answer. 
Otherwise, you need to output the complete answer that you think is correct following the train of thought of the existing steps.
Each step generated should be concise and focused, addressing only a small part of the solution. Avoid making the steps too complex or combining multiple ideas into one.
The complete solution should consist of at least three steps, so don't skip any essential steps.
Your output should be clear and systematic, with each step described one at a time to ensure logical progression.
Note: You are only allowed to describe the reasoning steps in natural language. Do not output any code. 
**If your answer includes code, it will cause unforeseen losses!**

**Output format:**
Step 1:
Step 2:
...
Step n:
Step n + 1:
Step n + 2:
...
Among them, Step 1 to Step n are consistent with the existing steps. Continue to generate based on the existing steps to obtain a complete answer.
The following is the input. Please output according to the specified output format, do not output unnecessary information, and do not repeat the question.
Input:
Problem:{problem}
Wrong code and reflections:{wrong_code_with_reflection}
Existing steps:{existing_steps}
'''

extract_step_prompt_en = '''I'll provide you with a complete answer, and your task is to break down each solution step. Wrap each solution step with the <step></step> XML tag.

**Input format (n steps):**
Complete answer:
Step 1:...
Step 2:...
...
Step n:...
Where "..." denotes omitted input information.

**Output format:**
<step>Step 1:...</step>
<step>Step 2:...</step>
...
<step>Step n:...</step>
Where "..." denotes omitted input information.
The following is the input. Please output according to the specified output format, do not output unnecessary information, and do not repeat the question.
Complete answer:'''

# 模拟阶段生成代码prompt
generate_code_prompt_en = '''You will play the role of a code implementer, writing a complete code based on the given problem and the step-by-step analysis of the problem. 
Your code must strictly follow the analysis steps provided and should not include your own opinions.
Rules:
- Importing function libraries(like: import math) and output function code only, without main function so that I can call your generated functions directly.
- The output code should be wrapped with code blocks (like ```python). Example: ```python\\ndef add(a, b):\\n    return a + b\\n```.

question:
{question}
analysis:
{analysis}
'''

# 扩展阶段生成下一步骤，带反思
code_proposal_prompt_use_reflection_en = '''
Your task is to provide the correct **next step** based on the previous incorrect code used to solve the problem and a reflection, for a given programming problem and its existing solution steps (which are incomplete).  
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

# 扩展阶段生成Step1，参考知识库Rag
code_proposal_prompt_use_knowledgebase_en = '''
Your task is to provide the correct **next step** based on the previous incorrect code used to solve the problem and a reflection, for a given programming problem and its existing solution steps (which are incomplete).  
Let's think step by step. But you only generate one step at a time.
We aim to decompose complex problems into a series of simpler subproblems and sequentially generate the corresponding steps to solve each subproblem. 
All the substeps should be combined in a way that avoids contradictions, forming a coherent solution to the original complex problem.

- The steps you generate will be passed to a code generation model, so they should be structured in a way that is easy for the model to understand.  
- Keep each step concise and focused, avoiding the inclusion of too much information at once. Ensure clear organization and logical progression in your reasoning.  
- **Important:** You can use very little code as detailed explanations in your answers, but you cannot just write code.
- **If your answer includes code, it will cause unforeseen losses!**
- Your answer should be based on the given analysis. Only if the analysis is wrong can you answer it in your own way.

- If no existing steps are provided, you should output the first step based on the given analysis.  
- If there are existing steps, output the next step (Step n+1) that logically follows the provided analysis and the previous steps.

The following content wrapped in <content> tags is to retrieve similar questions and solutions in the knowledge base. Refer to it to generate the next step if you think it is useful, otherwise ignore it directly.
{rag_content}

**Output format:**
Next step: ...

Where "..." is the next reasoning step you should fill in. This should be a clear and complete reasoning step, possibly including calculations, analysis, or decision-making.

**Here is the input. Please follow the restricted output format.**
Problem:
{problem}
'''

# 模拟阶段评估full_step
rollout_full_steps_prompt_en = '''Python code problem:
{problem}
Thoughts:
{solution}
Code:
{code}
Execute results on test cases:
{sandbox_result}

The above is a Python code problem, which includes the thoughts and code to solve the problem, as well as the return results of the code in the example test cases. In the return result, the status code 0 indicates that the test case has passed, -1 indicates timeout, and -2 indicates various other errors. Even if this code can pass all the example test cases, however, it may be correct or not completely correct.

Please evaluate and return the correctness score in range [0, 10.0]

Evaluate the correctness of the code and give only ONE evaluation score. 
The code's correctness is whether it can pass all the possible unseen test cases of the problem, not just the given ones.
When all the test cases can be passed, please pay strict attention to whether the possible boundary cases in the problem can be solved.
The test cases are correct. Please conduct reverse reasoning based on the test cases to understand the function signature. When you think the test cases are incorrect, please analyze the input and output of the test cases, and understand this code problem again. Analyze where there are problems in the steps of the problem-solving ideas, and think about how to modify these steps of the problem-solving ideas in order to pass the test cases.
Note: The solution should not modify the function signature or the test cases. If there is a contradiction between the function signature and the test cases, please study and understand the input and output formats of the test cases, and provide the correct problem-solving ideas based on the test cases.
Note: Your output should first analyze the overall code, and then assign a score. The score should be wrapped with the <evaluation></evaluation> tag.

Example Answers:
The code is far from correct for solving the problem.<evaluation>0.0</evaluation>   
The code can pass most test cases while may fail on some corner cases.<evaluation>8.5</evaluation>
I'll analyze the code for the `similar_elements` function to determine its correctness.\n\nThe function is designed to find shared elements between two tuples. Let's evaluate:\n\n1. **Approach**: The solution correctly uses set intersection to find common elements between the two tuples. This is an efficient approach with O(n+m) time complexity where n and m are the lengths of the tuples.\n\n2. **Implementation**: \n   - The code converts both tuples to sets\n   - Uses the & operator to find the intersection\n   - Converts the result back to a tuple\n   - Returns the resulting tuple of shared elements\n\n3. **Correctness**:\n   - The function correctly identifies common elements between tuples\n   - All test cases pass\n   - The implementation handles duplicates correctly (since sets remove duplicates)\n   - Edge cases like empty tuples would be handled correctly\n\n4. **Edge Cases**:\n   - Empty tuples: Would return an empty tuple\n   - One empty tuple: Would return an empty tuple\n   - No common elements: Would return an empty tuple\n   - Tuples with duplicates: Would return unique common elements\n\n5. **Potential Issues**: \n   - The function doesn't preserve the order of elements from the original tuples, but this doesn't seem to be a requirement based on the test cases\n   - The function doesn't preserve duplicates (if element appears multiple times in both tuples), but this seems acceptable given the use of sets and the test cases\n\nThe solution is correct, efficient, and handles all reasonable test cases for this problem statement. It uses the appropriate Python data structures and operations for the task.\n\n<evaluation>10.0</evaluation>
I'll analyze the code for the "Comp_Graph" problem to determine its correctness.\n\nThe problem requires assigning weights to erased edges in an undirected graph such that the shortest path between vertices s and t is exactly L. If this is impossible, we should output "NO".\n\nIssues with the current implementation:\n\n1. **Logic error in weight assignment**: The approach attempts to find a path and then distribute weights, but the implementation has several flaws:\n   - The code removes edges from `erased_edges_in_path` without properly tracking which one is the "last" edge\n   - When multiple erased edges exist in the path, it assigns weight 1 to all but "the last" edge, but doesn\'t correctly handle the final weight calculation\n\n2. **Path finding issues**: The current path finding algorithm might not find the optimal path to distribute weights.\n\n3. **Edge case handling**: The code fails on the test cases, which suggests it doesn\'t correctly handle the edge cases.\n\n4. **Test case failures**: All but one of the example test cases fail, indicating fundamental issues with the algorithm.\n\n5. **Inefficient approach**: The current approach tries to find a specific path and then assign weights, rather than considering the problem more holistically.\n\nA better approach would be:\n- Run Dijkstra\'s with all erased edges set to a very high value to find the shortest path using only non-erased edges\n- If this path length > L, then it\'s impossible to achieve L (return "NO")\n- If this path length == L, we can assign very large weights to all erased edges (return "YES")\n- If this path length < L, run Dijkstra\'s with all erased edges set to 1 to find the minimum possible path length\n- If this minimum path length > L, then it\'s impossible to achieve L (return "NO")\n- Otherwise, we need to find a specific path where we can distribute the remaining weight among erased edges\n\nThe current implementation has the right general idea but fails in the implementation details, particularly in weight assignment.\n\n<evaluation>2.5</evaluation>

Output:
'''

# 根据ldb_debug的反馈，指出错误的步骤和分析
remove_wrong_steps_prompt_en = '''Python code problem:
{problem}
Thoughts:
{solution}
Code:
{code}
Test case debug information:
{LDB_result}

The above is a Python code problem, which includes the thoughts and code for solving the problem, as well as the return results of debugging for a failed test case. If the debugging is successful, The debugging process is to first split the code into block - level code according to the AST. If the block - level code is correct after debugging analysis, the "correct" field is True, otherwise it is False. The "explanation" field is the analysis of the block - level code debugging.

Your task is to determine which specific step is written incorrectly based on the debug return results and conduct an analysis and summary. The correctly generated code and corresponding thought processes will be retained, while the incorrect code and corresponding thought processes will be discarded. You need to analyze and summarize the points to note so that subsequent thought processes can be generated based on the correct thought processes to correct the previous errors.

Your output consists of two parts:
1. Which specific step went wrong. Wrap it with the <step_n>x</step_n> XML tag, where x represents the specific number of the first erroneous step. If there are multiple erroneous steps in the thought process, only output the number of the first erroneous step. Do not output any extra content.
2. Analyze and summarize the points to note.

**Output format**
<step_n>x</step_n>...
Among them, ... represents the generated analysis.

Output:
'''

# ldb_debug失败，根据打分模型的分析结果指出错误的步骤和分析
remove_wrong_steps_debug_failed_prompt_en = '''Python code problem:
{problem}
Thoughts:
{solution}
Code:
{code}
reflection:
{reflection}

The above is a Python code problem, which includes the thoughts and code for solving the problem, as well as the reflection on the thoughts.

Your task is to determine which specific step is written incorrectly based on the reflection and conduct an analysis and summary.The correctly generated code and corresponding thought processes will be retained, while the incorrect code and corresponding thought processes will be discarded. You need to analyze and summarize the points to note so that subsequent thought processes can be generated based on the correct thought processes to correct the previous errors.

Your output consists of two parts:
1. Which specific step went wrong. Wrap it with the <step_n>x</step_n> XML tag, where x represents the specific number of the first erroneous step. If there are multiple erroneous steps in the thought process, only output the number of the first erroneous step. Do not output any extra content.
2. Analyze and summarize the points to note.

**Output format**
<step_n>x</step_n>...
Among them, ... represents the generated analysis.

Output:
'''

# 对已有步骤进行打分
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
