from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import traceback
import requests_cache
requests_cache.install_cache('jina_cache', expire_after=3600*24*365)
import time
import threading
import collections
import requests
import random
import urllib3

from typing import List, Dict

app = Flask(__name__)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
serper_keys = ["Place your own serper api key here"]

search_requests = collections.deque()
content_requests = collections.deque()

def count_recent_requests(request_queue, seconds=60):
    """计算最近指定秒数内的请求数量"""
    current_time = time.time()
    # 移除过期的请求记录
    while request_queue and request_queue[0] < current_time - seconds:
        request_queue.popleft()
    return len(request_queue)

def monitor_requests():
    """每10秒显示近一分钟内的请求数量"""
    while True:
        search_count = count_recent_requests(search_requests)
        content_count = count_recent_requests(content_requests)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 最近一分钟: search请求: {search_count}, get_content请求: {content_count}")
        time.sleep(10)

# 启动监控线程
monitor_thread = threading.Thread(target=monitor_requests, daemon=True)
monitor_thread.start()

import json
@app.route('/search', methods=['POST'])
def search():
    # 记录请求时间戳
    search_requests.append(time.time())
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': '请求中需要包含query参数'}), 400
        
        index = data.get('index', 'moegirl')
        query = data.get('query')
        error = False
        SOURCE = data.get('source', "googleapis")
        if 1:
            for i in range(6):
                try:
                    if SOURCE == "googleapis":
                        url = f"https://discoveryengine.googleapis.com/v1/projects/{project_id}/locations/global/collections/default_collection/engines/{app_ids[index]}/servingConfigs/default_search:searchLite?key={api_key}"
                        headers = {
                            "Content-Type": "application/json"
                        }
                        data = {
                            "servingConfig": f"projects/{project_id}/locations/global/collections/default_collection/engines/{app_ids[index]}/servingConfigs/default_search",
                            "query": query
                        }
                        
                        response = requests.post(url, headers=headers, json=data)
                        response_data = response.json()
                        if response.status_code == 200:
                            error = False
                            break
                        elif response.status_code == 429:
                            error = True
                            print(response.status_code)
                            time.sleep(2**(i+1))
                        else:
                            raise Exception(f"Google Search API 请求失败: {response.status_code}")
                    else:
                        try:
                            url = "https://google.serper.dev/search"
                            payload = json.dumps({
                                "q": query
                            })
                            headers = {
                                'X-API-KEY': random.choice(serper_keys),
                                'Content-Type': 'application/json'
                            }

                            response = requests.request("POST", url, headers=headers, data=payload)
                            response_data = response.json()
                            if response.status_code == 200:
                                error = False
                                break
                            #print(response.text)
                        except Exception as e:
                            if "429" in str(e) or "Quota exceed" in str(e):
                                print(429)
                                time.sleep(2**(i+1))
                                error = True
                            else:
                                raise e

                    # try:
                    #     response_data = search_with_client(project_id, "global", app_ids[index], query)
                    #     error = False
                    #     break
                    # except Exception as e:
                    #     if "429" in str(e) or "Quota exceed" in str(e):
                    #         print(429)
                    #         time.sleep(2**(i+1))
                    #         error = True
                    #     else:
                    #         raise e

 
                except Exception as e:
                    #traceback.print_exc()
                    print(e)
                    print(query)
                    error = True
            index = "moegirl"
            
            if not error:
                results = []
                if SOURCE == "googleapis":
                    for item in response_data.get('results', []):
                        document = item.get('document', {})
                        derived_data = document.get('derivedStructData', {})
                            
                        result_item = {
                            'id': document.get('id', ''),
                            'title': derived_data.get('title', ''),
                            'highlight': {'content': [derived_data.get('snippets', [])[0].get('htmlSnippet', '')]},
                            'url': derived_data.get('formattedUrl', '')
                        }
                            
                        results.append(result_item)
                else:
                    for id, item in enumerate(response_data.get('organic', [])):
                        result_item = {
                            'id': id,
                            'title': item.get('title', ''),
                            'highlight': {'content': [item.get('snippet', '')]},
                            'url': item.get('link', '')
                        }
                        results.append(result_item)
                return jsonify({'hits': results, 'success': True}), 200
        return jsonify({
            'success': False,
            'error': 'Search failed'
        }), 500
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
from fanoutqa.utils import markdownify
@app.route('/get_content', methods=['POST'])
def get_content():
    content_requests.append(time.time())
    
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': '请求中需要包含url参数'}), 400
    url = data['url']

    for i in range(3):
        url = "https://r.jina.ai/" + url
        headers = {
            #"Authorization": "Bearer " + random.choice(keys)
            "X-Md-Link-Style": "discarded",
            "X-Retain-Images": "none"
        }
        try:
            response = requests.get(url, headers=headers, proxies={'http': 'http://127.0.0.1:34523', 'https': 'http://127.0.0.1:34523'})
        except Exception as e:
            print(e)
            return jsonify({'error': '获取内容失败'}), 500
        if response.status_code == 200:
            return jsonify({
                'success': True,
                'content': response.text
            }) 
        else:
            print(response.text)
            if response.status_code == 429:
                time.sleep(random.randint(5, 30))
                #return jsonify({'error': '获取内容失败'}), 500
            else:
                return jsonify({'error': '获取内容失败'}), 500
    return jsonify({'error': '获取内容失败'}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=21021, debug=True)
