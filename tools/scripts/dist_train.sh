#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`
CONFIG=$1
NGPUS=$2
PY_ARGS=${@:3}
PROJ_ROOT='/mnt/csi-data-aly/user/hongfeizhang/mypaper/SparseFormerV2'
export PYTHONPATH=${PYTHONPATH}:${PROJ_ROOT}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} \
    ${PROJ_ROOT}/tools/train.py --launcher pytorch --cfg_file ${CONFIG} ${PY_ARGS} \
    --output_dir '/mnt/csi-data-aly/user/hongfeizhang/mypaper/SparseFormerV2/workdir/fshnet/encoder/fshnet_SPEncoder_3layer_1lc_zhx_.0diff_dense_hm_.3lr_.5wd' \
    --root_dir=${PROJ_ROOT} \
    --sync_bn \
    --wo_gpu_stat \
    --dataset='nuscenes' \
    2>&1 | tee log.train.$T
    # --dataset='waymo'/'nuscenes'/'argo2'
