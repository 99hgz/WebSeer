# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

import requests
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
GAME_SERVER_URL = "http://localhost:21022"
SEARCH_SERVER_URL = "http://localhost:21021"
class GuessTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        seed = kwargs["seed"]

        self._instance_dict[instance_id] = {
            "response": "",
            "times":0,
            "is_success": False,
        }

        response = requests.get(f"{GAME_SERVER_URL}/init?seed={seed}&instance_id={instance_id}")
        if response.status_code == 200:
            data = response.json()
            return data["prompt"]
        else:
            return f"初始化游戏失败"
            

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        character_name = parameters.get("character_name", "")
        response = requests.post(
            f"{GAME_SERVER_URL}/guess",
            json={"instance_id": instance_id, "character_name": character_name}
        )
        self._instance_dict[instance_id]["times"] += 1
        if response.status_code == 200:
            data = response.json()
            if data["status"] == 'success':
                self._instance_dict[instance_id]["is_success"] = True
            if 'prompt' not in data:
                return f"猜测角色失败: 角色不存在", 0.0, {}
            return data["prompt"], 0.0, {}
        else:
            return f"猜测角色失败: 角色不存在", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        if self._instance_dict[instance_id]["is_success"]:
            return 1.0 + 1/(self._instance_dict[instance_id]["times"] + 1)
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

import json
class SearchTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "times":0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        query = parameters.get("query", "")
        index = "moegirl"
        index = "web"
        response = requests.post(
            f"{SEARCH_SERVER_URL}/search",
            json={"query": query, "index": index}
        )
        self._instance_dict[instance_id]["times"] += 1
        if response.status_code == 200:
            # 格式化搜索结果为易读形式
            hits = response.json()['hits']
            formatted_results = []
            
            for idx, hit in enumerate(hits):
                title = hit.get("title", "无标题")
                url = hit.get("url", "")
                highlights = hit.get("highlight", {}).get("content", [])
                
                formatted_hit = f"{idx+1}. 标题: {title}\n链接url: {url}\n"
                if highlights:
                    formatted_hit += "内容摘要:\n"
                    for i, highlight in enumerate(highlights, 1):
                        formatted_hit += f" {highlight.strip()}\n"
                
                formatted_results.append(formatted_hit)
            
            # 将结果存储在实例字典中
            #self._instance_dict[instance_id]["response"] = formatted_results
            #return json.dumps(response.json()['hits'], ensure_ascii=False), 0.0, {}
            return ('\n'.join(formatted_results)) + '\n\n请先总结返回结果中有用的信息，再说明思路，以上信息即将消失。', 0.0, {}
        else:
            return f"搜索失败: 角色不存在", 0.0, {}
    

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        if self._instance_dict[instance_id]["times"] == 0:
            return -1.0
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
from openai import OpenAI

class QueryOnPageTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.client = OpenAI(api_key="sk-G7mmGTeHBJV4Wt79iksIUv4TnQbfpdATbnvtLQajLJRgjOmX", base_url="http://t.acac.ac.cn:26300/v1/")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "times":0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        url = parameters.get("url", "")
        try:
            response = requests.post(
                f"{SEARCH_SERVER_URL}/get_content",
                json={"url": url}
            )
            if response.status_code != 200:
                return f"获取页面内容失败", 0.0, {}
            self._instance_dict[instance_id]["times"] += 1
            content = response.json()['content']
            question = parameters.get("question", "")
            try:
                llm_response = self.client.chat.completions.create(
                    model="Qwen/Qwen3-30B-A3B",
                    messages=[
                        {"role": "system", "content": "你是一个帮助分析内容并回答问题的助手。请根据提供的内容回答问题。 /no_think"},
                        {"role": "user", "content": f"内容：{content}\n\n问题：{question}"}
                    ],
                    max_tokens=2048
                )
                return llm_response.choices[0].message.content, 0.0, {}
            except Exception as e:
                if "less than" in str(e):
                    print(e)
                    print("retry...")
                    content = content[:80000]
                    llm_response = self.client.chat.completions.create(
                        model="Qwen/Qwen3-30B-A3B",
                        messages=[
                            {"role": "system", "content": "你是一个帮助分析内容并回答问题的助手。请根据提供的内容回答问题。 /no_think"},
                            {"role": "user", "content": f"内容：{content}\n\n问题：{question}"}
                        ],
                        max_tokens=2048
                    )
                    return llm_response.choices[0].message.content, 0.0, {}
                else:
                    raise e
            
        except Exception as e:
            print(e)
            return f"查询失败", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]


class GetContentTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "times":0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        url = parameters.get("url", "")
        response = requests.get(
            f"{SEARCH_SERVER_URL}/get_content",
            json={"url": url}
        )
        self._instance_dict[instance_id]["times"] += 1
        if response.status_code == 200:
            return response.json()['content']
        else:
            return f"获取页面内容失败: {response.text}"

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["times"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
