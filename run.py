import subprocess
import sys
import time
import argparse
import os

def start_manage_server():
    """启动管理服务器"""
    print("启动管理服务器...")
    server_process = subprocess.Popen(
        [sys.executable, "manage_server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    # 等待服务器启动
    time.sleep(2)
    if server_process.poll() is not None:
        print("管理服务器启动失败!")
        stdout, stderr = server_process.communicate()
        print("标准输出:", stdout)
        print("错误输出:", stderr)
        sys.exit(1)
    
    return server_process

def run_simulation(num_workers=1):
    """运行多个模拟实例"""
    workers = []
    for i in range(num_workers):
        print(f"启动模拟实例 {i+1}...") 
        worker = subprocess.Popen(
            [sys.executable, "sim2.py", 'standard','--model', 'gpt-4.1-mini'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        workers.append(worker)
    
    return workers

def main():
    parser = argparse.ArgumentParser(description="运行猜角色模拟系统")
    parser.add_argument("--workers", type=int, default=1, help="并行运行的模拟实例数量")
    parser.add_argument("--server-only", action="store_true", help="只启动管理服务器")
    parser.add_argument("--worker-only", action="store_true", help="只启动模拟实例")
    args = parser.parse_args() 
    
    # 确保在正确的目录中
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    server_process = None
    workers = []
    
    try:
        # 启动管理服务器
        if not args.worker_only:
            server_process = start_manage_server()
            print("管理服务器已启动")
        
        # 启动模拟实例
        if not args.server_only:
            workers = run_simulation(args.workers)
            print(f"已启动 {len(workers)} 个模拟实例")
        time.sleep(3)
        
        # 检查是否有异常退出的进程
        while True:
            for i, worker in enumerate(workers):
                if worker.poll() is not None:
                    stdout, stderr = worker.communicate()
                    print(f"模拟实例 {i+1} 异常退出，返回码: {worker.returncode}")
                    # if stdout:
                    #     print(f"模拟实例 {i+1} 标准输出: {stdout}")
                    if stderr:
                        print(f"模拟实例 {i+1} 错误输出: {stderr}")
                    
                    # 重启异常退出的进程
                    print(f"重启模拟实例 {i+1}...")
                    workers[i] = subprocess.Popen(
                        [sys.executable, "sim2.py", 'standard','--model', 'gpt-4.1-mini'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(1)
     
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止所有进程...")
    
    finally:
        # 清理进程
        for i, worker in enumerate(workers):
            if worker.poll() is None:
                print(f"停止模拟实例 {i+1}...")
                worker.terminate()
                stdout, stderr = worker.communicate()
                if stderr:
                    print(f"模拟实例 {i+1} 错误输出:", stderr)
        
        if server_process and server_process.poll() is None:
            print("停止管理服务器...")
            server_process.terminate()
            stdout, stderr = server_process.communicate()
            if stderr:
                print("管理服务器错误输出:", stderr)

if __name__ == "__main__":
    main() 