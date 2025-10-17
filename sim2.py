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

client = OpenAI(api_key="", base_url="")
client_main = client
# 服务器配置
MANAGE_SERVER_URL = "http://localhost:21023"
SEARCH_SERVER_URL = "http://localhost:21021"

# 配置logging
# logging.basicConfig(
#     level=logging.ERROR,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         #logging.FileHandler('rag_simulation.log')
#     ]
# )
# logger = logging.getLogger(__name__)
# 自定义Logger类，将所有日志输出到print
class PrintLogger:
    def __init__(self, name):
        self.name = name
    
    def _log(self, level, message, *args, **kwargs):
        pass
        #print(f"[{level}] {self.name}: {message}")
    
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



#SUBMIT_MODE = "self-verify" / "multiple-times" / "default"
SUBMIT_MODE = "self-verify"
# 工具定义
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

def sample_verify_process(data, a, f1, model, max_turns = 10):
    q = data['question']
    messages = [
        {"role": "system", "content": """You are a reasoning assistant with the ability to perform web searches and execute Python code to help you process the content of the page and answer the question accurately.
Do not use any knowledge you know; all facts in your thinking should be obtained from the information returned by the tools. You can repeat the search process multiple times if necessary. 
Once you have all the information you need, continue your reasoning.
You should provide your final answer to the "submit_answer" tool. /no_think
"""},
        {"role": "user", "content": f"Please verify if the answer of question '{q}' is '{a}'. You can choose your answer from 'Correct', 'Partly Correct' or 'Incorrect'. You should provide your final answer to the 'submit_answer' tool."}
    ]
    for turn in range(max_turns):
        logger.info(f"\n sample_verify_process --- 第 {turn+1} 轮 ---")
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
        
        # 检查是否有工具调用
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            # 传递data参数，而不依赖全局变量
            tool_results = handle_tool_calls('', assistant_message.tool_calls, model, data, return_answer = True)
            
            # 添加工具结果到消息历史
            messages.extend(tool_results)
            
            # 检查是否猜对了
            for result in tool_results:
                if "submit_answer" in result["name"]:
                    result_data = result["content"]
                    result_data_code = 0
                    ground_truth_code = 0
                    if ('incorrect' in result_data.lower()):
                        result_data_code = 0
                    elif ('partly correct' in result_data.lower()):
                        result_data_code = 1
                    else:
                        result_data_code = 2
                    if (f1<=0.2):
                        ground_truth_code = 0
                    elif (f1>=0.8):
                        ground_truth_code = 2
                    else:
                        ground_truth_code = 1
                    #logger.debug(messages)
                    logger.info('sample_verify_process %s %s', result_data_code, ground_truth_code)
                    if (ground_truth_code == 0 and result_data_code in [0,1]) or (result_data_code == ground_truth_code):
                        return messages, True
                    else:
                        return messages, False
        else:
            # 如果没有工具调用，提示LLM使用工具
            messages.append({
                "role": "user",
                "content": "Please use the provided tools to answer the question."
            })
    
    logger.info("\n达到最大轮次限制，游戏结束。")
    return messages, False

def submit_answer(answer, model, data):
    if SUBMIT_MODE == "self-verify":
        retrys = 0
        if data['golden_answers'][0].lower() in answer.lower():
            f1 = 1.0
        else:
            f1 = calculate_f1(answer, data['golden_answers'][0])
        while retrys <= 10:
            logger.info('Start sample_verify_process', data['question'], data['golden_answers'][0], answer, f1)
            messages, success = sample_verify_process(data, answer, f1, model)
            # 检查是否使用了超过一次search
            search_count = 0
            for msg in messages:
                if msg["role"] == "assistant" and msg["tool_calls"]:
                    for tool_call in msg["tool_calls"]:
                        if tool_call["function"]["name"] == "search":
                            search_count += 1
            
            if search_count == 0:
                success = False

            if success:
                return messages
            else:
                retrys += 1
        raise Exception("Failed to verify the answer.")
    else:
        # 使用传入的data参数而不是全局变量
        if data['golden_answers'][0].lower() in answer.lower():
            return "Correct!"
        else:
            f1 = calculate_f1(answer, data['golden_answers'][0])
            return "Incorrect! The f1 score is " + str(f1) + ". Please first verify that the current result is incorrect, then continue trying to find the answer."

def search(query, index="web"):
    response = requests.post(
        f"{SEARCH_SERVER_URL}/search",
        json={"query": query, "index": index}
    )
    
    if response.status_code == 200:
        hits = response.json()['hits']
        formatted_results = []
            
        for idx, hit in enumerate(hits):
                title = hit.get("title", "Untitled")  # Changed from "无标题"
                url = hit.get("url", "")
                highlights = hit.get("highlight", {}).get("content", [])
                
                formatted_hit = f"{idx+1}. Title: {title}\nURL: {url}\n"  # Changed from "标题:" and "链接url:"
                if highlights:
                    formatted_hit += "Content Summary:\n"  # Changed from "内容摘要:"
                    for i, highlight in enumerate(highlights, 1):
                        formatted_hit += f" {highlight.strip()}\n"
                
                formatted_results.append(formatted_hit)
            
            # 将结果存储在实例字典中
            #self._instance_dict[instance_id]["response"] = formatted_results
            #return json.dumps(response.json()['hits'], ensure_ascii=False), 0.0, {}
        # The following prompt is already in English.
        return ('\n'.join(formatted_results)) + '\n\nPlease summarize the useful information in the returned results first, then explain your thinking. The above information will disappear soon.'
    else:
        return "Search failed: The specified item or index was not found." # Changed from "搜索失败: 角色不存在"
        
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
                model="Qwen/Qwen3-30B-A3B",
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
                    model="Qwen/Qwen3-30B-A3B",
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
            if return_answer:
                result = arguments["answer"]
            else:
                result = submit_answer(arguments["answer"], model, data)
            results.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": result,
                "signal": data['golden_answers'][0].lower() in arguments["answer"].lower() 
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
    """运行完整的猜角色模拟"""
    # 初始化游戏
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
        #logger.debug(messages)
        assistant_message = response.choices[0].message
        messages.append({
            "role": assistant_message.role,
            "content": assistant_message.content,
            "tool_calls": [_.to_dict() for _ in assistant_message.tool_calls] if assistant_message.tool_calls else [],
            'reasoning': assistant_message.reasoning_content if hasattr(assistant_message, 'reasoning_content') else "\n\n"
        })
        
        # 检查是否有工具调用
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            tool_results = handle_tool_calls(instance_id, assistant_message.tool_calls, model, data)
            
            # 添加工具结果到消息历史
            messages.extend(tool_results)
            
            # 检查是否猜对了
            for result in tool_results:
                if "submit_answer" in result["name"]:
                    if SUBMIT_MODE == "self-verify":
                        dialogs = result["content"][2:]
                        #logger.debug(result["content"])
                        for _ in tool_results:
                            messages.pop()
                        lastmsg = messages.pop()
                        tool_call_info = lastmsg["tool_calls"]
                        if lastmsg["content"] == None:
                            lastmsg["content"] = ''
                        lastmsg["content"] += 'So the answer is ' + json.loads(tool_call_info[0]["function"]["arguments"])["answer"]+'. '
                        lastmsg["content"] += "Wait, I need to verify the answer before submitting it." + (dialogs[0]["content"] if dialogs[0]["content"] else "")
                        lastmsg["tool_calls"] = dialogs[0]["tool_calls"]
                        lastmsg["reasoning"] = dialogs[0]["reasoning"]
                        messages.append(lastmsg)
                        dialogs.pop()
                        if result["signal"] == True:
                            dialogs[-1]["content"] = "So the answer is " + json.loads(tool_call_info[0]["function"]["arguments"])["answer"]
                            dialogs[-1]["tool_calls"] = tool_call_info
                        else:
                            dialogs[-1]["content"] = "So the current answer is " + ["Incorrect!", "Partly Correct!", "Correct!"][result["signal"]]
                            dialogs[-1]["tool_calls"] = []
                        messages.extend(dialogs[1:])
                        if result["signal"] == True:
                            return messages, True
                    else:
                        result_data = result["content"]
                        if "Correct!" in result_data:
                            logger.info(f"\n猜对了！")
                            return messages, True
                        else:
                            logger.info(f"\n猜错了！")
                            if SUBMIT_MODE == "default":
                                return messages, False

        else:
            # 如果没有工具调用，提示LLM使用工具
            messages.append({
                "role": "user",
                "content": "Please use the provided tools to answer the question."
            })
    
    return messages, False

def rewrite_simulation(data, max_turns=20, model="Qwen/Qwen3-32B", initial_messages=None):
    instance_id = str(uuid.uuid4())
    prompt = data['question']
    print(f"游戏ID: {instance_id}")
    print(f"初始提示: {prompt}")
    print(f"使用模型: {model}")

    messages = initial_messages
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg["content"]
            pattern = r'The markdown of page content was saved in variable page\d+\. You can access it by Python code\. And the answer to your question is: '
            msg["content"] = re.sub(pattern, '', content)
    pre_messages = []
    for msg in messages:
        if msg['role'] == 'assistant':
            if_submit = any(tool_call["function"]["name"] == 'submit_answer' for tool_call in msg["tool_calls"])
            if if_submit:
                cur_answer = ''
                for tool_call in msg["tool_calls"]:
                    if tool_call["function"]["name"] == 'submit_answer':
                        cur_answer = json.loads(tool_call["function"]["arguments"])["answer"]
                msg['content'] += 'So the answer is ' + cur_answer + '. Wait, I need to verify the answer before submitting it.'
                msg['tool_calls'] = []
                pre_messages.append(msg)

                while True:
                    for turn in range(max_turns):
                        logger.info(f"\n--- 第 {turn+1} 轮 ---")
                        
                        # 获取LLM回复
                        response = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=tools,
                            tool_choice="auto"
                        )
                        
                        assistant_message = response.choices[0].message
                        messages.append({
                            "role": assistant_message.role,
                            "content": assistant_message.content,
                            "tool_calls": [_.to_dict() for _ in assistant_message.tool_calls] if assistant_message.tool_calls else [],
                            'reasoning': assistant_message.reasoning_content
                        })
                        
                        # 检查是否有工具调用
                        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                            tool_results = handle_tool_calls(instance_id, assistant_message.tool_calls)
                            
                            # 添加工具结果到消息历史
                            messages.extend(tool_results)
                            
                            # 检查是否猜对了
                            for result in tool_results:
                                if "submit_answer" in result["name"]:
                                    result_data = result["content"]
                                    if "Correct!" in result_data:
                                        logger.info(f"\n猜对了！")
                                        return messages, True
                                    else:
                                        logger.info(f"\n猜错了！")
                                        #return messages, False

                        else:
                            # 如果没有工具调用，提示LLM使用工具
                            messages.append({
                                "role": "user",
                                "content": "Please use the provided tools to answer the question."
                            })

            else:
                pre_messages.append(msg)
        elif msg['role'] == 'user':
            pre_messages.append(msg)
        elif msg['role'] == 'tool':
            if msg['name'] != 'submit_answer':
                pre_messages.append(msg)
    
    
    logger.info("\n达到最大轮次限制，游戏结束。")
    return messages, False

def get_task_from_manager(worker_id):
    """从管理服务器获取一个任务（未完成的seed）"""
    try:
        response = requests.get(f"{MANAGE_SERVER_URL}/get_task", params={"worker_id": worker_id})
        if response.status_code == 200:
            data = response.json()
            return data.get("seed"), data.get("attempt_id"), data.get("worker_id")
        else:
            logger.error(f"获取任务失败: {response.text}")
            return None, None, None 
    except Exception as e:
        logger.error(f"获取任务时出错: {str(e)}")
        return None, None, None

def report_task_result(seed, messages, success, attempt_id=None, worker_id=None, model=None):
    """向管理服务器报告任务结果"""
    try:
        payload = {
            "seed": seed,
            "success": success,
            "messages": messages,
            "attempt_id": attempt_id,
            "worker_id": worker_id,
            "model": model
        }
        response = requests.post(f"{MANAGE_SERVER_URL}/report_result", json=payload)
        if response.status_code == 200:
            logger.info(f"结果上报成功: seed={seed}, success={success}")
            return True
        else:
            logger.error(f"结果上报失败: {response.text}")
            return False
    except Exception as e:
        logger.error(f"上报结果时出错: {str(e)}")
        return False

def process_continuation(item, datas, model):
    """处理单个继续执行的任务"""
    try:
        seed = item["seed"]
        initial_messages = item["messages"]
        currect_data = datas[seed]
        
        # 继续执行现有的messages
        messages, success = run_simulation(currect_data, max_turns=10, model=model, initial_messages=initial_messages)
        
        # 返回结果
        return {
            "seed": seed,
            "messages": messages,
            "success": success
        }
    except Exception as e:
        logger.error(f"处理任务时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "seed": item.get("seed", "unknown"),
            "messages": item.get("messages", []),
            "success": False,
            "error": str(e)
        }

def process_offline_task(seed, datas, model, max_turns=10):
    """处理单个离线任务"""
    try:
        data = datas[seed]
        logger.info(f"处理任务: seed={seed}")
        # 不再设置全局变量，直接传递data参数
        messages, success = run_simulation(data, max_turns=max_turns, model=model)
        return {
            "messages": messages,
            "seed": seed,
            "success": success
        }
    except Exception as e:
        logger.error(f"处理任务时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "seed": seed,
            "messages": [],
            "success": False,
            "error": str(e)
        }

def run_standard_mode(model="Qwen/Qwen3-235B-A22B", max_turns=10, offline=False, parallel=4):
    """运行标准模式，从服务器获取任务"""
    worker_id = str(uuid.uuid4())
    datas = []
    with jsonlines.open('/home/hgz/data/first10k.jsonl') as reader:
        for line in reader:
            datas.append(line)
    
    print(f"使用模型: {model}")
    
    if offline:
        print(f"运行离线模式，测试seed 1-500")
        seeds = list(range(1, 501))
        
        # 创建处理函数的偏函数，固定datas和model参数
        process_func = partial(process_offline_task, datas=datas, model=model, max_turns=max_turns)
        
        # 创建进程池并使用imap处理任务
        with Pool(parallel) as pool:
            results = []
            success_count = 0
            total_processed = 0
            
            # 使用tqdm显示进度，并添加实时成功率显示
            pbar = tqdm(total=len(seeds), desc="离线测试进度")
            for result in pool.imap(process_func, seeds):
                results.append(result)
                total_processed += 1
                if result.get("success", False):
                    success_count += 1
                
                # 更新进度条，显示当前成功率
                success_rate = (success_count / total_processed) * 100 if total_processed > 0 else 0
                pbar.set_postfix(成功率=f"{success_count}/{total_processed} ({success_rate:.2f}%)")
                pbar.update(1)
            
            pbar.close()
        
        # 统计最终成功率
        print(f"测试完成。成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.2f}%)")
        
        # 保存结果到JSON文件
        output_path = f"offline_results_{model.replace('/', '_')}_{int(time.time())}.json"
        print(f"保存结果到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
    else:
        global currect_data  # 非离线模式下仍然使用全局变量，因为是单线程
        while True:
            #try:
            if 1:
                seed, attempt_id, worker_id = get_task_from_manager(worker_id)
                if seed is None:
                    logger.info("没有获取到任务，等待1秒后重试...")
                    time.sleep(1)
                    continue
                    
                print(f"获取到任务: seed={seed}, attempt_id={attempt_id}, worker_id={worker_id}")
                currect_data = datas[seed]
                messages, success = run_simulation(currect_data, max_turns=max_turns, model=model)
                print('===== final messages =====')
                #logger.debug(messages)
                # 向管理服务器报告结果
                report_task_result(seed, messages, success, attempt_id, worker_id, model)
                
            # except Exception as e:
            #     logger.error(f"运行过程中出错: {str(e)}")
            #     logger.error(traceback.format_exc())
            time.sleep(1)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RAG问答系统运行脚本')
    
    # 添加子命令解析器
    subparsers = parser.add_subparsers(dest='mode', help='运行模式')
    
    # 标准模式参数
    standard_parser = subparsers.add_parser('standard', help='标准模式：从服务器获取任务并执行')
    standard_parser.add_argument('--model', type=str, default="Qwen/Qwen3-235B-A22B", help='使用的模型名称')
    standard_parser.add_argument('--max-turns', type=int, default=10, help='最大对话轮次')
    standard_parser.add_argument('--offline', action='store_true', help='是否使用离线模式测试seed 1-500')
    standard_parser.add_argument('--parallel', type=int, default=4, help='并行处理的数量（仅离线模式有效）')
    
    # 继续执行模式参数
    continue_parser = subparsers.add_parser('continue', help='继续执行模式：从JSON文件加载数据并继续执行')
    continue_parser.add_argument('--input', type=str, default="multiple_submit_attempts.json", help='输入JSON文件路径')
    continue_parser.add_argument('--output', type=str, default=f"continuation_results_{int(time.time())}.json", help='输出JSON文件路径')
    continue_parser.add_argument('--parallel', type=int, default=4, help='并行处理的数量')
    continue_parser.add_argument('--model', type=str, default="Qwen/Qwen3-235B-A22B", help='使用的模型名称')
    continue_parser.add_argument('--max-turns', type=int, default=10, help='最大对话轮次')
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有指定模式，默认为标准模式
    if args.mode is None:
        args.mode = 'standard'
        
    return args

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 根据模式执行相应的功能
    if args.mode == 'continue':
        print(f"使用继续执行模式")
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"并行度: {args.parallel}")
        print(f"使用模型: {args.model}")
        print(f"最大轮次: {args.max_turns}")
        
        run_standard_mode(
            model=args.model,
            max_turns=args.max_turns,
            offline=args.offline,
            parallel=args.parallel
        )
    else:  # 标准模式
        print(f"使用标准模式")
        print(f"使用模型: {args.model}")
        print(f"最大轮次: {args.max_turns}")
        if args.offline:
            print(f"离线模式: 是")
            print(f"并行度: {args.parallel}")
        
        run_standard_mode(
            model=args.model,
            max_turns=args.max_turns,
            offline=args.offline,
            parallel=args.parallel
        )

