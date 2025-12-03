"""
本地服务启动示例
CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen2.5-7B-Instruct --gpu_memory_utilization 0.9 --max_model_len 8000 --dtype=half > vllm.log 2>&1 &
"""

import os
import backoff
import openai
from openai import OpenAI
from dotenv import load_dotenv


# Load environment variables from .env file
env_file_path = os.path.join(os.path.dirname(__file__), '../../configs/.env')
if not os.path.exists(env_file_path):
    print(f"Warning: {env_file_path} does not exist. Please check the path.")
load_dotenv(env_file_path)


# 重试装饰器
def create_openai_backoff():
    return backoff.on_exception(
        backoff.constant,
        openai.OpenAIError,
        max_tries=20,  # 最大重试次数
        interval=10,    # 重试间隔时间
        on_backoff=lambda details: print(f"chat_models_api.py: 错误信息: {details['exception']} 重试中... ({details['tries']}次)"),
    )

class ChatAPI:
    def __init__(self, model_name=None, api_key=None, api_base=None):
        self.model_name = model_name or os.environ["MODEL_NAME"]
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
        self.api_base = api_base or os.environ["OPENAI_API_BASE"]
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        self.completion_tokens = 0
        self.prompt_tokens = 0
    
    @create_openai_backoff()
    def _generate_openai(self, messages, model, **kwargs) -> list:
        res = self.client.chat.completions.create(
            messages=messages,
            model=model, 
            **kwargs,
        )
        outputs = [choice.message.content for choice in res.choices]

        # Log completion tokens
        self.completion_tokens += res.usage.completion_tokens
        self.prompt_tokens += res.usage.prompt_tokens

        return outputs

    def generate(self, prompt, temperature=0.7, **kwargs) -> list:
        if "gemini" in self.model_name:
            return self._generate_gemini(prompt, temperature, **kwargs)
        elif "qwen3" in self.model_name:
            # kwargs中必须包含extra_body={"enable_thinking": False}
            kwargs["extra_body"] = {"enable_thinking": False}
            messages = [{"role": "user", "content": prompt}]
            return self._generate_openai(messages, model=self.model_name, temperature=temperature, **kwargs)
        else:
            messages = [{"role": "user", "content": prompt}]
            return self._generate_openai(messages, model=self.model_name, temperature=temperature, **kwargs)

    def generate_chat(self, messages, temperature=0.7, **kwargs) -> list:
        if "qwen3" in self.model_name:
            # kwargs中必须包含extra_body={"enable_thinking": False}
            kwargs["extra_body"] = {"enable_thinking": False}
            return self._generate_openai(messages, model=self.model_name, temperature=temperature, **kwargs)
        else:
            return self._generate_openai(messages, model=self.model_name, temperature=temperature, **kwargs)

    def token_usage(self):
        return {"completion_tokens": self.completion_tokens, "prompt_tokens": self.prompt_tokens}
    
    def get_client_model_list(self):
        models = self.client.models.list()
        print("Available models:")
        for model in models.data:
            print(f"- {model.id}")
        return [model.id for model in models.data]

    # new add
    @create_openai_backoff()
    def _generate_gemini(self, prompt, temperature):
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=self.model_name, 
                    temperature=temperature,
                    base_url=self.api_base,
                    api_key=self.api_key)
        res = llm.invoke(prompt)
        return [res.content]


if __name__ == '__main__':
    # 测试接口
    model = ChatAPI(
        # model_name="Qwen2.5-14B-Instruct",
        # api_key="EMPTY",
        # api_base="http://localhost:8000/v1"
    )
    prompt = "你好！能为我介绍一下Python编程语言吗？"
    output_list = model.generate(prompt, temperature=0.7, max_tokens=100, n=1)
    print(output_list[0])
    print(model.token_usage())