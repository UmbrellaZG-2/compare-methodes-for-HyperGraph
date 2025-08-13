import os
import subprocess
import sys
import os
import glob

# 定义项目路径和对应的主脚本及参数
projects = [
    {
        "name": "HCoN",
        "script": "run.py",
        "param_name": "--dataname"
    },
    {
        "name": "HGNN",
        "script": "hgnn_time.py",
        "param_name": "--dataname"
    },
    {
        "name": "HNHN",
        "script": "run_HNHN.py",
        "param_name": "--dataset_name"
    },
    {
        "name": "UniGNN",
        "script": "train.py",
        "param_name": "--dataset",
        "extra_params": ["--data", "custom"]
    }
]

# 定义数据文件夹和结果文件夹路径
data_dir = os.path.abspath("data")
result_dir = os.path.abspath("result")

# 确保data和result文件夹存在
os.makedirs(data_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# 获取data目录下所有.mat文件
data_files = glob.glob(os.path.join(data_dir, "*.mat"))

if not data_files:
    print("警告: 未在data目录下找到任何.mat文件，程序退出。")
    sys.exit(1)

# 为每个数据集执行所有项目
for data_file in data_files:
    # 提取文件名（不含扩展名）
    dataset_name = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n开始处理数据集: {dataset_name}")

    for project in projects:
        project_name = project["name"]
        print(f"  开始执行项目: {project_name}")
        project_dir = os.path.join(os.getcwd(), project_name)

        # 检查项目目录是否存在
        if not os.path.exists(project_dir):
            print(f"  警告: 项目 {project_name} 的目录不存在，跳过执行。")
            continue

        # 构建执行命令
        script_path = os.path.join(project_dir, project["script"])
        cmd = [sys.executable, script_path]

        # 添加额外参数（如果有）
        if "extra_params" in project:
            cmd.extend(project["extra_params"])

        # 添加数据集参数
        cmd.extend([project["param_name"], dataset_name])

        # 添加GPU参数（可选，根据需要调整）
        if project_name == "HCoN" or project_name == "HGNN" or project_name == "HNHN":
            cmd.extend(["--gpu_id", "0"])
        elif project_name == "UniGNN":
            cmd.extend(["--gpu", "0"])

        # 执行命令
        try:
            print(f"  执行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=project_dir)
            print(f"  项目 {project_name} 执行完成。")
        except subprocess.CalledProcessError as e:
            print(f"  项目 {project_name} 执行失败: {e}")

print("\n所有项目和数据集执行完毕。")