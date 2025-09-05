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
        "gpu_param": "--gpu_id"
    },
    {
        "name": "LEGCN",
        "script": "main.py",
        "param_name": "--dataset",
        "gpu_param": "--gpu"
    },
    {
        "name": "UniGNN",
        "script": "train.py",
        "param_name": "--dataset",
        "gpu_param": "--gpu"
    },
    {
        "name": "SCN",
        "script": "run_ours.py",
        "param_name": "--dataname",
        "gpu_param": "--gpu_id"
    }
]

data_dir = os.path.abspath("data")
result_dir = os.path.abspath("result")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# 只处理指定的三个数据集
specified_datasets = [
        "dblp_co_I",
        "dblp_co_II",
        "citation_co_I",
        "citation_co_II",
        "acm_co_I",
        "acm_co_II",
    ]
data_files = []

for dataset in specified_datasets:
    file_path = os.path.join(data_dir, f"{dataset}.mat")
    if os.path.exists(file_path):
        data_files.append(file_path)
    else:
        print(f"警告: 数据集文件 {dataset}.mat 不存在")

if not data_files:
    print("警告: 未找到指定的数据集文件，程序退出。")
    sys.exit(1)

for data_file in data_files:
    dataset_name = os.path.splitext(os.path.basename(data_file))[0]
    # 对于acm_co_I_10%、acm_co_I_20%、acm_co_I_30%，直接使用原始名称
    config_dataset_name = dataset_name
    
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

        cmd.extend([project["param_name"], config_dataset_name])
        cmd.extend([project["gpu_param"], str(gpu_id)])

        try:
            print(f"  执行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=project_dir)
            print(f"  项目 {project_name} 执行完成。")
        except subprocess.CalledProcessError as e:
            print(f"  项目 {project_name} 执行失败: {e}")

print("\n所有项目和数据集执行完毕。")