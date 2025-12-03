import dataclasses
from typing import *
from .tracing import get_code_traces_block, get_code_traces_line, get_code_traces_function


IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str

def print_messages(messages: List[Message], prefix = "") -> None:
    print("===CHAT MESSAGE===" + f"({prefix})")
    for msg in messages:
        print(msg.content)
    print("==================")

def messages_to_dicts(messages: List[Message]) -> List[dict]:
    return [{"role": message.role, "content": message.content} for message in messages]


def ldb_debug(prompt: str, prev_func_impl: str, failed_test: str, entry: str, model: Any, messages: Any, dataset_type: str = "", level: str = "block") -> str:
        print("Start ldb_debug...")
        failed_test_string = failed_test.split("# Real Execution Output:")[0]
        real_test_output = failed_test.split("# Real Execution Output:")[1]
        if len(messages) == 0:
            messages = [
                Message(
                    role = "system",
                    content = "You are an expert programming assistant.",
                ),
                Message(
                    role = "user",
                    content = f"Complete the following task in Python. Please respond with code only (with the code inside a Markdown code block).\n{prompt}"
                ),
                Message(
                    role = "assistant",
                    content = f"{prev_func_impl}"
                )
            ]
        feedback = f"The code above fails the given unit test:\n{failed_test}. \nHelp me debug this.\n"
        # Check whether the solution can be executed
        if level == "line":
            trace_blocks = get_code_traces_line(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
        elif level == "function":
            trace_blocks = get_code_traces_function(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
        else:
            trace_blocks = get_code_traces_block(IMPORT_HEADER + prev_func_impl, failed_test.replace("assert ", "").split("==")[0], entry)
        print("Get trace blocks...")
        # CANNOT EXECUTED
        if isinstance(trace_blocks, str):
            if trace_blocks == "*timeout*":
                print("The program exceeds the time limit!")
                msg = [Message(role = "user", content = f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
            elif trace_blocks.startswith("*execution fail*"):
                print(trace_blocks.replace("*execution fail*", ""))
                msg = [Message(role = "user", content = f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
            elif trace_blocks.startswith("*parse fail*"):
                print("The program is weird")
                msg = [Message(role = "user", content = f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
            else:
                assert False, "Strange type of trace error: " + trace_blocks
            print_messages(msg, "execute error")
            messages += msg
            return {
                "is_success": False,
                "result": f"Execute error. {msg[0].content}",
            }
        elif len(trace_blocks) == 0:
            print("No trace blocks found.")
            msg = [Message(role = "user", content = f"Feedback: With the above function, the assertion is `{failed_test_string}` but the real execution output is `{real_test_output}`.")]
            print_messages(msg, "No trace blocks")
            messages += msg
            return {
                "is_success": False,
                "result": f"No trace blocks found. {msg[0].content}",
            }
        # Start debugging
        msg = [Message(
                    role = "user",
                    content = feedback + "\nHere is the code execution trace block by block with the intermediate variable values. Please explain the execution FOR EACH BLOCK and answer whether this block is correct or not. If not, give an explanation on what is wrong. Please wrap your response into a JSON object that contains keys `block` with the name of each block, key `correct` with value False or True, and key `explanation` with an explanation on the bug. \nExample Answers:\n{\"block\": \"BLOCK-1\", \"correct\": \"True\", \"explanation\": \"The block initializes variable `a` and `b`.\"}\n{\"block\": \"BLOCK-2\", \"correct\": \"False\", \"explanation\": \"The block is incorrect because the code does not add the two integers together, but instead subtracts the second integer from the first. To fix this issue, we should change the operator from `-` to `+` in the return statement. This will ensure that the function returns the correct output for the given input.\"}"
            )]
        
        if level == "line":
            max_num_blocks = 30
        elif level == "function":
            max_num_blocks = 1
            block_lines = trace_blocks[0]
            print("313:", len(block_lines))
            if len(block_lines) > 30:
                trace_blocks[0] = block_lines[:15] + ["..."] + block_lines[-15:]
        else:
            max_num_blocks = 10
        if len(trace_blocks) > max_num_blocks:
            print("Sample trace block...")
            selected_blocks = trace_blocks[:int(max_num_blocks/2)] + trace_blocks[-int(max_num_blocks/2):]
            trace_blocks  = selected_blocks
        result = ""
        for i, b in enumerate(trace_blocks):
            b = "\n".join(b)
            b = f"\n[BLOCK-{i}]\n" + b
            msg[0].content += b
            result += b
        msg[0].content += "\n"
        messages += msg
        # print_messages(messages, "prompt")
        outputs = model.generate_chat(
            messages=messages_to_dicts(messages),
            temperature=0,
            stop=['[debug end]', 'Here is the updated code:']
        )
        explanation_all = outputs[0]
        result = result + '\n' + explanation_all

        #wrong_block, explanation = parse_explanation(explanation_all, trace_blocks, prev_func_impl)
        msg = [
            Message(
                    role = "assistant",
                    content = explanation_all
                )
        ]
        print_messages(msg, "response")
        messages += msg
        return {
            "is_success": True,
            "result": result,
        }

