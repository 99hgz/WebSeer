from flask import Flask, jsonify, request, render_template
import sqlite3
import random
import json
import os
import datetime
import uuid

app = Flask(__name__)

# 配置
DATABASE_PATH = "tasks.db"
PORT = 21023
HOST = "0.0.0.0"

# 初始化数据库
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # 创建任务表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        seed INTEGER PRIMARY KEY,
        status TEXT DEFAULT 'pending',
        attempts INTEGER DEFAULT 0,
        last_attempt TIMESTAMP,
        success BOOLEAN DEFAULT 0
    )
    ''')
    
    # 创建任务尝试表 - 存储每个worker的尝试记录
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS task_attempts (
        attempt_id TEXT PRIMARY KEY,
        seed INTEGER,
        worker_id TEXT,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        status TEXT DEFAULT 'running',
        success BOOLEAN DEFAULT 0,
        messages TEXT,
        model TEXT DEFAULT NULL,
        FOREIGN KEY (seed) REFERENCES tasks (seed)
    )
    ''')
    
    # 检查是否需要初始化任务
    cursor.execute("SELECT COUNT(*) FROM tasks")
    count = cursor.fetchone()[0]
    
    if count == 0:
        # 创建一些初始任务
        for seed in range(1, 1001):  # 创建100个初始种子
            cursor.execute("INSERT OR IGNORE INTO tasks (seed) VALUES (?)", (seed,))
    
    conn.commit()
    conn.close()
    print("数据库初始化完成")

# API路由
@app.route('/get_task', methods=['GET'])
def get_task():
    # 获取worker ID，如果没有提供则生成一个
    worker_id = request.args.get('worker_id', str(uuid.uuid4()))
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # 获取所有未成功完成的任务
    cursor.execute("""
    SELECT seed FROM tasks 
    WHERE success = 0 
    ORDER BY attempts ASC, RANDOM() 
    LIMIT 1
    """)
    
    task = cursor.fetchone()
    
    if task:
        seed = task[0]
        # 生成尝试ID
        attempt_id = str(uuid.uuid4())
        
        # 更新任务表中的尝试次数和最后尝试时间
        cursor.execute("""
        UPDATE tasks 
        SET attempts = attempts + 1, 
            last_attempt = CURRENT_TIMESTAMP,
            status = 'running'
        WHERE seed = ?
        """, (seed,))
        
        # 在尝试表中添加新记录
        cursor.execute("""
        INSERT INTO task_attempts 
        (attempt_id, seed, worker_id, start_time, status) 
        VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'running')
        """, (attempt_id, seed, worker_id))
        
        conn.commit()
        conn.close()
        return jsonify({"seed": seed, "attempt_id": attempt_id, "worker_id": worker_id})
    else:
        conn.close()
        return jsonify({"error": "没有可用的任务"}), 404

@app.route('/report_result', methods=['POST'])
def report_result():
    data = request.json
    seed = data.get('seed')
    attempt_id = data.get('attempt_id')
    success = data.get('success', False)
    messages = json.dumps(data.get('messages', []), ensure_ascii=False)
    worker_id = data.get('worker_id')
    model = data.get('model', None)  # 获取模型信息
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # 更新任务尝试记录
    cursor.execute("""
    UPDATE task_attempts 
    SET status = ?, 
        success = ?, 
        messages = ?,
        model = ?,
        end_time = CURRENT_TIMESTAMP
    WHERE attempt_id = ?
    """, ("completed" if success else "failed", success, messages, model, attempt_id))
    
    # 如果成功，更新主任务状态
    if success:
        cursor.execute("""
        UPDATE tasks 
        SET status = 'completed', 
            success = 1
        WHERE seed = ?
        """, (seed,))
    # else:
    #     # 检查是否有其他正在运行的尝试
    #     cursor.execute("""
    #     SELECT COUNT(*) FROM task_attempts 
    #     WHERE seed = ? AND status = 'running'
    #     """, (seed,))
        
    #     running_count = cursor.fetchone()[0]
        
    #     if running_count == 0:
    #         # 没有正在运行的尝试，将任务状态设为failed
    #         cursor.execute("""
    #         UPDATE tasks 
    #         SET status = 'failed'
    #         WHERE seed = ? AND success = 0
    #         """, (seed,))
    
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success"})

@app.route('/add_tasks', methods=['POST'])
def add_tasks():
    data = request.json
    start_seed = data.get('start_seed', 1)
    end_seed = data.get('end_seed', 100)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    for seed in range(start_seed, end_seed + 1):
        cursor.execute("INSERT OR IGNORE INTO tasks (seed) VALUES (?)", (seed,))
    
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success", "added_tasks": end_seed - start_seed + 1})

@app.route('/reset_task/<int:seed>', methods=['POST'])
def reset_task(seed):
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # 重置主任务状态
    cursor.execute("""
    UPDATE tasks 
    SET status = 'pending', 
        attempts = 0, 
        success = 0, 
        last_attempt = NULL
    WHERE seed = ?
    """, (seed,))
    
    # 可选：保留历史尝试记录，但将其标记为已重置
    cursor.execute("""
    UPDATE task_attempts 
    SET status = 'reset'
    WHERE seed = ?
    """, (seed,))
    
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success"})

@app.route('/status', methods=['GET'])
def status():
    # 获取状态筛选参数
    status_filter = request.args.get('status', 'all')
    
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 获取任务统计信息
    cursor.execute("""
    SELECT 
        COUNT(*) as total, -- 总尝试次数
        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending, -- 'pending' 状态的尝试次数 (通常为0，因为尝试记录创建时状态为 'running')
        SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running, -- 'running' 状态的尝试次数
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed, -- 'completed' 状态的尝试次数
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed, -- 'failed' 状态的尝试次数
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful -- 成功的尝试次数
    FROM task_attempts
    """)
    stats = dict(cursor.fetchone())
    
    # 构建查询语句，根据筛选条件过滤
    query = """
    SELECT 
        ta.attempt_id, 
        ta.seed, 
        ta.worker_id,
        ta.status, 
        ta.start_time, 
        ta.end_time,
        ta.success,
        ta.model,
        t.attempts
    FROM task_attempts ta
    JOIN tasks t ON ta.seed = t.seed
    """
    
    # 添加筛选条件
    if status_filter != 'all':
        query += f" WHERE ta.status = '{status_filter}'"
    
    # 添加排序
    query += " ORDER BY ta.start_time DESC LIMIT 100"
    
    # 获取最近的任务尝试
    cursor.execute(query)
    
    recent_attempts = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return render_template('status.html', stats=stats, recent_attempts=recent_attempts, selected_status=status_filter)

@app.route('/view_attempt/<attempt_id>', methods=['GET'])
def view_attempt(attempt_id):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 获取尝试详情
    cursor.execute("""
    SELECT * FROM task_attempts WHERE attempt_id = ?
    """, (attempt_id,))
    
    attempt = dict(cursor.fetchone() or {})
    
    if attempt:
        # 解析消息JSON
        try:
            messages = json.loads(attempt.get('messages', '[]'))
        except:
            messages = []
    else:
        messages = []
    
    conn.close()
    
    return render_template('attempt_detail.html', attempt=attempt, messages=messages)

# 添加HTML模板
@app.route('/')
def index():
    return render_template('index.html')

# 创建模板目录和模板文件
def create_templates():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # 创建index.html
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>任务管理服务器</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        h1 { color: #333; }
        .container { max-width: 1200px; margin: 0 auto; }
        .btn { display: inline-block; padding: 8px 16px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG-RL 任务管理服务器</h1>
        <p>这是管理猜角色任务的服务器。</p>
        <a href="/status" class="btn">查看任务状态</a>
    </div>
</body>
</html>
        ''')
    
    # 创建status.html
    with open('templates/status.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>任务状态</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        h1, h2 { color: #333; }
        .container { max-width: 1200px; margin: 0 auto; }
        .stats { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }
        .stat-card { background: #f5f5f5; border-radius: 8px; padding: 20px; flex: 1; min-width: 150px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .stat-value { font-size: 24px; font-weight: bold; margin-top: 10px; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .success { color: green; }
        .failed { color: red; }
        .pending { color: orange; }
        .running { color: blue; }
        .btn { display: inline-block; padding: 8px 16px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px; margin-right: 10px; }
        .refresh { background: #2196F3; }
        .actions { margin: 20px 0; }
        .filter-section { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 8px; }
        .filter-btn { padding: 6px 12px; margin-right: 5px; background: #e0e0e0; border: none; border-radius: 4px; cursor: pointer; }
        .filter-btn.active { background: #2196F3; color: white; }
    </style>
    <meta http-equiv="refresh" content="10">
</head>
<body>
    <div class="container">
        <h1>任务状态</h1>
        
        <div class="actions">
            <a href="/" class="btn">返回首页</a>
            <a href="/status" class="btn refresh">刷新</a>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div>总任务数</div>
                <div class="stat-value">{{ stats.total }}</div>
            </div>
            <div class="stat-card">
                <div>待处理</div>
                <div class="stat-value">{{ stats.pending }}</div>
            </div>
            <div class="stat-card">
                <div>运行中</div>
                <div class="stat-value">{{ stats.running }}</div>
            </div>
            <div class="stat-card">
                <div>已完成</div>
                <div class="stat-value">{{ stats.completed }}</div>
            </div>
            <div class="stat-card">
                <div>失败</div>
                <div class="stat-value">{{ stats.failed }}</div>
            </div>
            <div class="stat-card">
                <div>成功率</div>
                <div class="stat-value">{{ "%.1f"|format(stats.successful / ((stats.successful + stats.failed + 1)) * 100) }}%</div>
            </div>
        </div>
        
        <div class="filter-section">
            <h3>筛选任务</h3>
            <a href="/status" class="filter-btn {{ 'active' if not status_filter else '' }}">全部</a>
            <a href="/status?status=completed" class="filter-btn {{ 'active' if status_filter == 'completed' else '' }}">已完成</a>
            <a href="/status?status=running" class="filter-btn {{ 'active' if status_filter == 'running' else '' }}">运行中</a>
            <a href="/status?status=pending" class="filter-btn {{ 'active' if status_filter == 'pending' else '' }}">待处理</a>
            <a href="/status?status=failed" class="filter-btn {{ 'active' if status_filter == 'failed' else '' }}">失败</a>
        </div>
        
        <h2>最近任务尝试</h2>
        <table>
            <thead>
                <tr>
                    <th>种子</th>
                    <th>尝试ID</th>
                    <th>Worker ID</th>
                    <th>模型</th>
                    <th>状态</th>
                    <th>开始时间</th>
                    <th>结束时间</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
                {% for attempt in recent_attempts %}
                <tr>
                    <td>{{ attempt.seed }}</td>
                    <td>{{ attempt.attempt_id[:8] }}...</td>
                    <td>{{ attempt.worker_id[:8] }}...</td>
                    <td>{{ attempt.model if attempt.model else '未知' }}</td>
                    <td class="{{ attempt.status }}">{{ attempt.status }}</td>
                    <td>{{ attempt.start_time }}</td>
                    <td>{{ attempt.end_time or '进行中' }}</td>
                    <td>
                        <a href="/view_attempt/{{ attempt.attempt_id }}">查看详情</a>
                        <form action="/reset_task/{{ attempt.seed }}" method="post" style="display: inline;">
                            <button type="submit">重置任务</button>
                        </form>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
        ''')
        
    # 创建attempt_detail.html
    with open('templates/attempt_detail.html', 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>尝试详情</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        h1, h2 { color: #333; }
        .container { max-width: 1200px; margin: 0 auto; }
        .detail-card { background: #f5f5f5; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .detail-item { margin-bottom: 10px; }
        .label { font-weight: bold; display: inline-block; width: 120px; }
        .success { color: green; }
        .failed { color: red; }
        .running { color: blue; }
        .btn { display: inline-block; padding: 8px 16px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px; margin-right: 10px; }
        .message-container { background: white; border: 1px solid #ddd; border-radius: 4px; padding: 10px; margin-bottom: 10px; }
        .message { margin-bottom: 15px; }
        .message-role { font-weight: bold; margin-bottom: 5px; }
        .user { color: #2196F3; }
        .assistant { color: #4CAF50; }
        .tool { color: #FF9800; }
        .system { color: #9C27B0; }
        pre { background: #f8f8f8; padding: 10px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>尝试详情</h1>
        
        <div class="actions">
            <a href="/status" class="btn">返回状态页面</a>
        </div>
        
        <div class="detail-card">
            <div class="detail-item">
                <span class="label">尝试ID:</span> {{ attempt.attempt_id }}
            </div>
            <div class="detail-item">
                <span class="label">种子:</span> {{ attempt.seed }}
            </div>
            <div class="detail-item">
                <span class="label">Worker ID:</span> {{ attempt.worker_id }}
            </div>
            <div class="detail-item">
                <span class="label">状态:</span> <span class="{{ attempt.status }}">{{ attempt.status }}</span>
            </div>
            <div class="detail-item">
                <span class="label">模型:</span> {{ attempt.model or "未知" }}
            </div>
            <div class="detail-item">
                <span class="label">成功:</span> {{ "是" if attempt.success else "否" }}
            </div>
            <div class="detail-item">
                <span class="label">开始时间:</span> {{ attempt.start_time }}
            </div>
            <div class="detail-item">
                <span class="label">结束时间:</span> {{ attempt.end_time or "进行中" }}
            </div>
            
        </div>
        
        <h2>对话记录</h2>
        <div class="message-container">
            {% for message in messages %}
                <div class="message">
                    <div class="message-role {{ message.role }}">{{ message.role }}</div>
                    {% if message.content %}
                        <div class="message-content">{{ '<think>' + message.reasoning + '</think>' if message.reasoning else '' }} {{ message.content }}</div>
                    {% endif %}
                    
                    {% if message.tool_calls %}
                        <div class="tool-calls">
                            <strong>工具调用:</strong>
                            <pre>{{ message.tool_calls|safe }}</pre>
                        </div>
                    {% endif %}
                </div>
            {% else %}
                <p>没有对话记录</p>
            {% endfor %}
        </div>
    </div>
</body>
</html>
        ''')

if __name__ == "__main__":
    # 初始化数据库
    init_db()
    
    # 创建模板
    create_templates()
    
    # 启动服务器
    print(f"管理服务器启动在 http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=True)
