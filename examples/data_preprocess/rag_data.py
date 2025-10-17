# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import time
import datasets
import requests
from verl.utils.hdfs_io import copy, makedirs
import jsonlines
import pandas as pd
import json
if __name__ == "__main__":
    datas = []
    if 0:
        with jsonlines.open('/home/hgz/data/first10k.jsonl') as reader:
            for line in reader:
                datas.append(line)
    else:
        file_path = "/data3/hgz/verl7/train.parquet"
        df = pd.read_parquet(file_path)
        gts = json.loads(df.to_json(orient="records", force_ascii=False))
        datas=[{
            'question': v['prompt'][0]['content'],
            'golden_answers': [v['reward_model']['ground_truth']],
            'metadata':{'level': 'hard'}
        } for v in gts]
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/re_rag_rl")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_list = [{"seed": i} 
                 for i in range(501, 10000) if 'metadata' in datas[i] and 'level' in datas[i]['metadata'] and datas[i]['metadata']['level'] == 'hard']
    
    train_size = int(0.8 * len(data_list))
    
    dataset = datasets.Dataset.from_list(data_list)
    dataset = dataset.train_test_split(train_size=train_size, shuffle=True, seed=42)
    
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    #instruction_following = "你是一个擅长猜测动漫角色的助手。你需要根据提供的描述，猜测出符合描述的角色，答案一定存在并且可以被找到。你可以使用工具来帮助你搜索信息和提交多次猜测。"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            seed = example["seed"]
            
            question = datas[seed]['question']
            answer = datas[seed]['golden_answers'][0]
            data = {
                "data_source": 'rag',
                "prompt": [
            {"role": "system", "content": """You are a reasoning assistant with the ability to perform web searches and execute Python code to help you process the content of the page and answer the user's question accurately.
Do not use any knowledge you know; all facts in your thinking should be obtained from the information returned by the tools. You can repeat the search process multiple times if necessary. 
Once you have all the information you need, continue your reasoning.
Please answer the following question. You should provide your final answer to the "submit_answer" tool.
"""},
            {"role": "user", "content": f"Question: {question}"}
                ],
                "ability": "rag",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                    "seed": seed,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "submit_answer": {
                            "create_kwargs": {"seed": seed, "ground_truth": answer},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                        "search": {
                            "create_kwargs": {"seed": seed},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                        "query_on_page": {
                            "create_kwargs": {"seed": seed},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            #time.sleep(1)
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    #test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
