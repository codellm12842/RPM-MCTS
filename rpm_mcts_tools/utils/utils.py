import os
import json
import re
from typing import List, Dict, Union


### 路径 ###
def get_root_path(dir_name='ours'):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    root_path = cur_path[:cur_path.find(dir_name)] + dir_name
    print(f"root_path: {root_path}")
    return root_path

### JSON文件读写 ###
def read_json(file_path: str) -> List[Dict]:
    assert os.path.exists(file_path), f"File not found: {file_path}"
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    print(f"Loaded {len(data)} items from {file_path}")
    return data

def write_json(data: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} items to {output_path}")

def write_jsonl(data: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} items to {output_path}")

def write_jsonl_append(data: Union[Dict, List[Dict]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if isinstance(data, dict):
        data = [data]
    with open(output_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Appended {len(data)} items to {output_path}")

### 提取模型输出 ###
def extract_output_from_llm_response(
    raw_output: str,
    tags_to_extract: List[str],
    has_think: bool = False,
) -> dict:
    result = {}
    if has_think:
        # 分割 think 和 answer 部分
        parts = raw_output.split("</think>", 1)
        think = parts[0].strip() if len(parts) > 1 else ""
        answer = parts[1].strip() if len(parts) > 1 else raw_output.strip()
        result["model_think"] = think
        result["model_answer"] = answer
    else:
        answer = raw_output

    for tag in tags_to_extract:
        # 提取标签内容，如果有多个匹配则取第一个
        pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
        match = pattern.search(answer)
        if match:
            content = match.group(1).strip()
        else:
            print(f"Warning: No match for tag <{tag}> in the response(length: {len(answer)}): {repr(answer[:100])}...")
            content = ""
        result[tag] = content
    return result

def extract_python_code(response: str) -> str:
    pattern = r'```python(.*?)```'
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()
    else:
        print("Warning: No code block found in the response, return response.")
        return response
