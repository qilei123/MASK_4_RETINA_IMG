export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
export PYTHONPATH=${PYTHONPATH}:../MASK_4_RETINA_IMG/incubator-mxnet/python/

TRAIN_DIR=/home/qileimail123/data0/RetinaImg/ROP_COCO/output/baseline2/
DATASET=coco
SET=train2014
TEST_SET=val2014
mkdir -p ${TRAIN_DIR}

# Train
python train_alternate_mask_fpn.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --pretrained model/resnet-50 \
    --prefix ${TRAIN_DIR} \
    --pretrained_epoch 0 \
    --gpu 1 & tee -a ${TRAIN_DIR}/train.log

