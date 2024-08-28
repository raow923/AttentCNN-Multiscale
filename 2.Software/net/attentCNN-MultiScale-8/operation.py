import subprocess

# 定义包含文件路径和tw值的列表
file_paths = ["./train.py", "./train.py", "./train.py", "./train.py", "./train.py"]
tw_values = [0.2, 0.4, 0.6, 0.8, 1.0]  # 你想要设置的tw值

# 依次执行列表中的每个文件
for file_path, tw_value in zip(file_paths, tw_values):
    print(f"Running {file_path} with tw={tw_value}")
    process = subprocess.Popen(["python", file_path, "--tw", str(tw_value)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 实时打印输出
    for line in iter(process.stdout.readline, ''):
        print(line, end='')

    # 等待进程结束并获取输出和错误信息
    stdout, stderr = process.communicate()
    if stderr:
        print(f"Errors in {file_path}:\n{stderr}")

