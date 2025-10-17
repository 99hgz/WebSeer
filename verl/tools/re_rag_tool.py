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
import asyncio
from typing import Any, Optional, Tuple
from uuid import uuid4

import requests
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
SEARCH_SERVER_URL = "http://localhost:21021"
REQUEST_TIMEOUT = 120  # 设置120秒超时
import re
import string
from collections import Counter

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def replace_hyphens(text):
        return text.replace("-", " ")
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(replace_hyphens(lower(s))))).strip()


def calculate_f1(prediction, gold):
    prediction = normalize_answer(prediction)
    gold = normalize_answer(gold)
        
    prediction_tokens = prediction.split()
    ground_truth_tokens = gold.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if (num_same == 0):  # avoid division by zero
        return 0
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def calculate_recall(prediction, gold):
    prediction = normalize_answer(prediction)
    gold = normalize_answer(gold)
        
    prediction_tokens = prediction.split()
    ground_truth_tokens = gold.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if (num_same == 0):  # avoid division by zero
        return 0
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return recall

class SubmitAnswerTool(BaseTool):

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "answer_times": 0,
            "correct": False,
            "ground_truth": ground_truth.lower(),
            "f1": 0.0,
            "recall": 0.0
        }
        return instance_id
        # seed = kwargs["seed"]


        # response = requests.get(f"{GAME_SERVER_URL}/init?seed={seed}&instance_id={instance_id}")
        # if response.status_code == 200:
        #     data = response.json()
        #     return data["prompt"]
        # else:
        #     return f"初始化游戏失败"
            

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        # character_name = parameters.get("character_name", "")
        # response = requests.post(
        #     f"{GAME_SERVER_URL}/guess",
        #     json={"instance_id": instance_id, "character_name": character_name}
        # )
        # self._instance_dict[instance_id]["times"] += 1
        # if response.status_code == 200:
        #     data = response.json()
        #     if data["status"] == 'success':
        #         self._instance_dict[instance_id]["is_success"] = True
        #     if 'prompt' not in data:
        #         return f"猜测角色失败: 角色不存在", 0.0, {}
        #     return data["prompt"], 0.0, {}
        # else:
        #     return f"猜测角色失败: 角色不存在", 0.0, {}
        answer = parameters.get("answer", "")
        # 确保answer是字符串类型
        if not isinstance(answer, str):
            try:
                answer = str(answer)
            except Exception as e:
                answer = ""
        self._instance_dict[instance_id]["answer_times"] += 1
        self._instance_dict[instance_id]["f1"] = calculate_f1(answer, self._instance_dict[instance_id]["ground_truth"])
        self._instance_dict[instance_id]["recall"] = max(self._instance_dict[instance_id]["recall"], calculate_recall(answer, self._instance_dict[instance_id]["ground_truth"]))
        self._instance_dict[instance_id]["correct"] = (self._instance_dict[instance_id]["recall"] > 0.9)
        if self._instance_dict[instance_id]["correct"]:
            return "Correct", self._instance_dict[instance_id]["answer_times"], {}
        else:
            return "Incorrect! The f1 score is " + str(self._instance_dict[instance_id]["recall"]) + ". Please continue trying to find the answer.", self._instance_dict[instance_id]["answer_times"], {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        correctness_reward = self._instance_dict[instance_id]["recall"]
        penalty = max(0, self._instance_dict[instance_id]["answer_times"] - 1)
        # if self._instance_dict[instance_id]["correct"]:
        #     return 1.0
        # if self._instance_dict[instance_id]["answered"]==False:
        #     return -1.0
        return correctness_reward * (0.8 ** penalty)
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
        self._instance_dict[instance_id]["times"] += 1
        query = parameters.get("query", "")
        index = "wiki"
        try:
            async def make_request():
                response = requests.post(
                    f"{SEARCH_SERVER_URL}/search",
                    json={"query": query, "index": index},
                    timeout=REQUEST_TIMEOUT
                )
                return response
            
            # 使用asyncio.wait_for包装请求，设置120秒超时
            response = await asyncio.wait_for(make_request(), timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                hits = response.json()['hits'][:5]
                formatted_results = []
                    
                for idx, hit in enumerate(hits):
                    title = hit.get("title", "Untitled")
                    url = hit.get("url", "")
                    highlights = hit.get("highlight", {}).get("content", [])
                    
                    formatted_hit = f"{idx+1}. Title: {title}\nURL: {url}\n" 
                    if highlights:
                        formatted_hit += "Content:\n"
                        for i, highlight in enumerate(highlights, 1):
                            formatted_hit += f" {highlight.strip()}\n"
                    
                    formatted_results.append(formatted_hit)
                    
                return ('\n'.join(formatted_results)) + '\n\nPlease summarize the useful information in the returned results first, then explain your thinking. The above information will disappear soon.', 0.0, {}
            else:
                return "Search failed: The specified item or index was not found.", 0.0, {}
        except asyncio.TimeoutError:
            logger.error("Search request timed out after %s seconds", REQUEST_TIMEOUT)
            return "Search failed: Request timed out", 0.0, {}
        except Exception as e:
            logger.error(str(e))
            return "Search failed", 0.0, {}
    

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
        self.client = OpenAI(api_key="sk-G7mmGTeHBJV4Wt79iksIUv4TnQbfpdATbnvtLQajLJRgjOmX", base_url="http://t.acac.ac.cn:26300/v1/", timeout=REQUEST_TIMEOUT)

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
        question = parameters.get("question", "")
        try:
            async def get_content():
                response = requests.post(
                    f"{SEARCH_SERVER_URL}/get_content",
                    json={"url": url},
                    timeout=REQUEST_TIMEOUT
                )
                return response
            
            # 使用asyncio.wait_for包装请求，设置120秒超时
            response = await asyncio.wait_for(get_content(), timeout=REQUEST_TIMEOUT)
            
            if response.status_code != 200:
                return "Failed to get page content", 0.0, {}
            
            content = response.json()['content']
            try:
                async def call_llm():
                    return self.client.chat.completions.create(
                        model="Qwen/Qwen3-30B-A3B",
                        messages=[
                            {"role": "system", "content": "You are an assistant that helps analyze content and answer questions. Please answer the questions based on the provided content. /no_think"},
                            {"role": "user", "content": f"Content: {content}\n\nQuestion: {question}"}
                        ],
                        max_tokens=2048
                    )
                
                # 使用asyncio.wait_for包装LLM调用，设置120秒超时
                llm_response = await asyncio.wait_for(call_llm(), timeout=REQUEST_TIMEOUT)
                return llm_response.choices[0].message.content, 0.0, {}
            except asyncio.TimeoutError:
                logger.error("LLM request timed out after %s seconds", REQUEST_TIMEOUT)
                return "Query failed: LLM request timed out", 0.0, {}
            except Exception as e:
                if "less than" in str(e):
                    logger.error(str(e))
                    logger.info("retry...")
                    content = content[:70000]
                    
                    async def call_llm_retry():
                        return self.client.chat.completions.create(
                            model="Qwen/Qwen3-30B-A3B",
                            messages=[
                                {"role": "system", "content": "You are an assistant that helps analyze content and answer questions. Please answer the questions based on the provided content, and do not fabricate information not present in the text. /no_think"},
                                {"role": "user", "content": f"Content: {content}\n\nQuestion: {question}"}
                            ],
                            max_tokens=2048
                        )
                    
                    # 使用asyncio.wait_for包装LLM调用，设置120秒超时
                    llm_response = await asyncio.wait_for(call_llm_retry(), timeout=REQUEST_TIMEOUT)
                    return llm_response.choices[0].message.content, 0.0, {}
                else:
                    raise e
        except asyncio.TimeoutError:
            logger.error("Request timed out after %s seconds", REQUEST_TIMEOUT)
            return "Query failed: Request timed out", 0.0, {}        
        except Exception as e:
            logger.error(str(e))
            return "Query failed", 0.0, {}

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
        try:
            async def get_content():
                response = requests.get(
                    f"{SEARCH_SERVER_URL}/get_content",
                    json={"url": url},
                    timeout=REQUEST_TIMEOUT
                )
                return response
            
            # 使用asyncio.wait_for包装请求，设置120秒超时
            response = await asyncio.wait_for(get_content(), timeout=REQUEST_TIMEOUT)
            
            self._instance_dict[instance_id]["times"] += 1
            if response.status_code == 200:
                return response.json()['content']
            else:
                return f"获取页面内容失败: {response.text}"
        except asyncio.TimeoutError:
            logger.error("Request timed out after %s seconds", REQUEST_TIMEOUT)
            return "获取页面内容失败: 请求超时", 0.0, {}
        except Exception as e:
            logger.error(str(e))
            return f"获取页面内容失败: {str(e)}", 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["times"]

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
