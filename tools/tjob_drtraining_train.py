from datetime import datetime
import os

NOW = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
TAG = "train"
if os.environ.get("TAG") is not None:
    TAG = os.environ.get("TAG")

job_name=f'nusc-3d-det-{TAG}-{NOW}'

deekeeper_project_name='nusc-3d-det'
deekeeper_experiment_name=job_name

code_dir="/mnt/csi-data-aly/user/hongfeizhang/mypaper/SparseFormerV2/tools"
config='cfgs/sparse_models/sparse_former_base.yaml'

priority=4  # 普通用户最高4，管理员用户最高8
name=job_name
num_machine=1 # 实例数
num_gpu=8   # 单实例GPU数
gpu_type='A30' # A30 or L20
num_cpu=96 # 单实例CPU数

image="master0:5000/sparseformerv2:v1.0"

platform='dr_training'
project="monorepo"

base_train_cmd = f"scripts/dist_train.sh {config} {num_gpu}"

print(base_train_cmd)

run = [
    "nvidia-smi",
    "conda init bash",
    "source /root/miniconda3/etc/profile.d/conda.sh",
    "conda activate sparseformerv2",
    f"cd {code_dir}",
    "sleep 2s",
    "pwd",
    "chmod +x scripts/dist_train.sh",
    base_train_cmd,
]
# description='tsr train test'
