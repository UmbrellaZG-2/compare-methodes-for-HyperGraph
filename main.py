import os
import subprocess
import sys
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

gpu_id = args.gpu

projects = [
    {
        "name": "HCoN",
        "script": "run.py",
        "param_name": "--dataname",
        "gpu_param": "--gpu"
    },
    {
        "name": "HGNN",
        "script": "hgnn_time.py",
        "param_name": "--dataname",
        "gpu_param": "--gpu"
    },
    {
        "name": "HNHN",
        "script": "run_HNHN.py",
        "param_name": "--dataset_name",
        "gpu_param": "--gpu"
    },
    {
        "name": "UniGNN",
        "script": "train.py",
        "param_name": "--dataset",
        "extra_params": ["--data", "custom"],
        "gpu_param": "--gpu"
    },
    {
        "name": "LEGCN",
        "script": "main.py",
        "param_name": "--dataset",
        "gpu_param": "--gpu"
    }
]

data_dir = os.path.abspath("data")
result_dir = os.path.abspath("result")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
data_files = glob.glob(os.path.join(data_dir, "*.mat"))

if not data_files:
    print("警告: 未在data目录下找到任何.mat文件，程序退出。")
    sys.exit(1)

for data_file in data_files:
    dataset_name = os.path.splitext(os.path.basename(data_file))[0]
    print(f"\n开始处理数据集: {dataset_name}")

    for project in projects:
        project_name = project["name"]
        print(f"  开始执行项目: {project_name}")
        project_dir = os.path.join(os.getcwd(), project_name)

        if not os.path.exists(project_dir):
            print(f"  警告: 项目 {project_name} 的目录不存在，跳过执行。")
            continue

        script_path = os.path.join(project_dir, project["script"])
        cmd = [sys.executable, script_path]

        if "extra_params" in project:
            cmd.extend(project["extra_params"])

        cmd.extend([project["param_name"], dataset_name])
        cmd.extend([project["gpu_param"], str(gpu_id)])

        try:
            print(f"  执行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=project_dir)
            print(f"  项目 {project_name} 执行完成。")
        except subprocess.CalledProcessError as e:
            print(f"  项目 {project_name} 执行失败: {e}")

print("\n所有项目和数据集执行完毕。")