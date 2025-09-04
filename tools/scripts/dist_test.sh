#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

CONFIG=$1
NGPUS=$2
CKPT=$3
PY_ARGS=${@:4}
PROJ_ROOT='/mnt/csi-data-aly/user/hongfeizhang/mypaper/SparseFormerV2'

export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=${NGPUS} ${PROJ_ROOT}/tools/test.py --launcher pytorch \
    --cfg_file ${CONFIG} \
    --ckpt ${CKPT} \
    --output_dir ' /mnt/csi-data-aly/user/hongfeizhang/mypaper/SparseFormerV2/workdir/voxelnext/encoder/voxelnext_en_3layer_1lc_zhx_de_1layer_v3_32p_.0diff_.1lr_.1wd' \
    --root_dir=${PROJ_ROOT} \
    --dataset='nuscenes' \
    ${PY_ARGS} 2>&1 | tee log.test.$T
