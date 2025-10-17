import requests
import json
import random
import time
from openai import OpenAI
import sys
import uuid
import traceback
import jsonlines
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import os
import argparse

#place your own api key and base url here

client = OpenAI(api_key="", base_url="http://127.0.0.1:20090/v1/")
client_main = client

SEARCH_SERVER_URL = "http://localhost:21021"

class PrintLogger:
    def __init__(self, name):
        self.name = name
    
    def _log(self, level, message, *args, **kwargs):
        pass
        print(f"[{level}] {self.name}: {message}")
    
    def debug(self, message, *args, **kwargs):
        self._log("DEBUG", message, *args, **kwargs)
    
    def info(self, message, *args, **kwargs):
        self._log("INFO", message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        self._log("WARNING", message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        self._log("ERROR", message, *args, **kwargs)
    
    def critical(self, message, *args, **kwargs):
        self._log("CRITICAL", message, *args, **kwargs)
    
    def exception(self, message, *args, **kwargs):
        exc_info = sys.exc_info()
        self._log("ERROR", message, *args, **kwargs)
        print(f"Exception: {exc_info[1]}")
        print(traceback.format_exc())

# 创建logger实例
logger = PrintLogger(__name__)

tools = [
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit your final answer. You must use this tool to submit your answer before the dialog ends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Your final answer"
                    }
                },
                "required": ["answer"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Call google to search for relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_on_page",
            "description": "This tool will visit a specific page of url, and it will answer the question based on the content of the page. The assistant has no context information, please describe the question completely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url of the page, must be a page provided by the search tool."
                    },
                    "question": {
                        "type": "string",
                        "description": "The question about the content of the page"
                    }
                },
                "required": ["url", "question"]
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "execute_python_code",
    #         "description": "Input ahe Python code will be executed by an external sandbox, and the output can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. ",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "code": {
    #                     "type": "string",
    #                     "description": "The Python code to be executed."
    #                 }
    #             },
    #             "required": ["code"]
    #         }
    #     }
    # }
    
]
global currect_data
import re
import string
from collections import Counter

def submit_answer(answer, model, data):
    pass

def search(query, index="web"):
    response = requests.post(
        f"{SEARCH_SERVER_URL}/search",
        json={"query": query, "index": index, "source": 'serper'}
    )
    
    if response.status_code == 200:
        hits = response.json()['hits']
        formatted_results = []
            
        for idx, hit in enumerate(hits):
                title = hit.get("title", "Untitled") 
                url = hit.get("url", "")
                highlights = hit.get("highlight", {}).get("content", [])
                
                formatted_hit = f"{idx+1}. Title: {title}\nURL: {url}\n" 
                if highlights:
                    formatted_hit += "Content Summary:\n"  
                    for i, highlight in enumerate(highlights, 1):
                        formatted_hit += f" {highlight.strip()}\n"

                formatted_results.append(formatted_hit)
            
        return ('\n'.join(formatted_results)) + '\n\nPlease summarize the useful information in the returned results first, then explain your thinking. The above information will disappear soon.'
    else:
        return "Search failed: The specified item or index was not found." 
        
def query_on_page(url, question):
    global global_vars1,global_vars_id
    try:
        response = requests.post(
            f"{SEARCH_SERVER_URL}/get_content",
            json={"url": url}
        )
        if response.status_code != 200:
            return f"Failed to get page content"
        
        content = response.json()['content']
        global_vars_id+=1
        global_vars1[f"page{global_vars_id}"] = content
        try:
            llm_response = client.chat.completions.create(
                model="WebSeer-14b",
                messages=[
                    {"role": "system", "content": "You are an assistant that helps analyze content and answer questions. Please answer the questions based on the provided content. /no_think"},
                    {"role": "user", "content": f"Content: {content}\n\nQuestion: {question}"}
                ],
                max_tokens=2048
            )
            return llm_response.choices[0].message.content
            return 'The markdown of page content was saved in variable page' + str(global_vars_id) + '. You can access it by Python code. And the answer to your question is: ' + llm_response.choices[0].message.content
        except Exception as e:
            if "less than" in str(e):
                logger.error(str(e))
                logger.info("retry...")
                content = content[:80000]
                llm_response = client.chat.completions.create(
                    model="WebSeer-14b",
                    messages=[
                        {"role": "system", "content": "You are an assistant that helps analyze content and answer questions. Please answer the questions based on the provided content, and do not fabricate information not present in the text. /no_think"},
                        {"role": "user", "content": f"Content: {content}\n\nQuestion: {question}"}
                    ],
                    max_tokens=2048
                )
                return llm_response.choices[0].message.content
                return 'The markdown of page content was saved in variable page' + str(global_vars_id) + '. You can access it by Python code. And the answer to your question is: ' + llm_response.choices[0].message.content
            else:
                raise e
        
    except Exception as e:
        logger.error(str(e))
        return f"Query failed"

from executor import *
global_vars1 = {}
global_vars_id = 0
def execute_python_code(code):
    executor = PythonExecutor()
    executor.runtime.inject(global_vars1)
    if code.strip() == "":
        return "error: empty code"
    try:
        result, report = executor.apply(code)
        if report == "Done":
            return result
        else:
            return f"error: {report}"
    except Exception as e:
        logger.error(traceback.format_exc())
        return f"error: {str(e)}"

def _safe_parse_args(raw):
    """Decode JSON once or twice until we get a dict."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
            if isinstance(raw, str):           # double-encoded?
                raw = json.loads(raw)
        except json.JSONDecodeError:
            pass                               # fall back to original
    return raw if isinstance(raw, dict) else {}

def handle_tool_calls(instance_id, tool_calls, model, data, return_answer = False):
    """处理工具调用"""
    results = []
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = _safe_parse_args(tool_call.function.arguments)
        #print(tool_call.function.arguments, arguments)
        if function_name == "submit_answer":
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": arguments["answer"]
            })
        
        elif function_name == "search":
            query = arguments["query"]
            index = 'wiki'
            result = search(query, index)
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": result
            })
        
        elif function_name == "query_on_page":
            url = arguments["url"]
            question = arguments["question"]
            try:
                result = query_on_page(url, question)
                results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": result
                })
            except Exception as e:
                results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(e)
                })
        elif function_name == "execute_python_code":
            code = arguments["code"]
            result = execute_python_code(code)
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": result
            })
    return results
from pprint import pprint
from copy import deepcopy
def run_simulation(data, max_turns=20, model="Qwen/Qwen3-32B", initial_messages=None):
    instance_id = str(uuid.uuid4())
    prompt = data['question']
    
    # 初始化对话或使用提供的messages继续
    if initial_messages:
        messages = initial_messages
    else:
        messages = [
            {"role": "system", "content": """You are a reasoning assistant with the ability to perform web searches and execute Python code to help you process the content of the page and answer the user's question accurately.
Do not use any knowledge you know; all facts in your thinking should be obtained from the information returned by the tools. You can repeat the search process multiple times if necessary. 
Once you have all the information you need, continue your reasoning.
Please answer the following question. You should provide your final answer to the "submit_answer" tool. /no_think
"""},
            {"role": "user", "content": f"Question: {prompt}"}
        ]
    
    # 对话轮次
    for turn in range(max_turns):
        logger.info(f"\n--- 第 {turn+1} 轮 ---")
        nmsg = deepcopy(messages)

        # 检查messages中的每条消息，如果tool_calls为空列表，则删除这个键
        for msg in nmsg:
            if "tool_calls" in msg and msg["tool_calls"] == []:
                del msg["tool_calls"]

        # 获取LLM回复
        response = client_main.chat.completions.create(
            model=model,
            messages=nmsg,
            tools=tools,
            tool_choice="auto"
        )
        assistant_message = response.choices[0].message
        messages.append({
            "role": assistant_message.role,
            "content": assistant_message.content,
            "tool_calls": [_.to_dict() for _ in assistant_message.tool_calls] if assistant_message.tool_calls else [],
            'reasoning': assistant_message.reasoning_content if hasattr(assistant_message, 'reasoning_content') else "\n\n"
        })
        logger.debug(messages)
        # 检查是否有工具调用
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            tool_results = handle_tool_calls(instance_id, assistant_message.tool_calls, model, data)
            
            # 添加工具结果到消息历史
            messages.extend(tool_results)
            
            for result in tool_results:
                if "submit_answer" in result["name"]:
                        return messages, True

        else:
            messages.append({
                "role": "user",
                "content": "Please use the provided tools to answer the question."
            })
    
    return messages, False

if __name__ == "__main__":
    q = "How many years earlier would Punxsutawney Phil have to be canonically alive to have made a Groundhog Day prediction in the same state as the US capitol?"
    messages, success = run_simulation({'question': q}, max_turns=20, model="WebSeer-14b")
    print(messages)

