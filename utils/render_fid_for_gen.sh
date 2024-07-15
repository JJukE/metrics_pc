export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/root/dev/scenegen/t2std/

# # NFD
# python scripts/TSD/metrics/utils/render_fid.py \
#     --gen_dir /root/hdd1/T2STD/exp/nfd \
#     --gen_out_dir /root/hdd1/T2STD/res/NFD/render_for_fid \
#     --num_views 20 \
#     --suffix nomtl.obj \
#     --type gen \
#     --process_with_model_id

# TSD
python scripts/TSD/metrics/utils/render_fid.py \
    --gen_dir /root/hdd1/t2std_a100/res/TSD_train_ids/240606_2333_generation \
    --gen_out_dir /root/hdd1/t2std_a100/res/TSD_train_ids/240606_2333_generation/render_for_fid \
    --num_views 20 \
    --suffix nomtl.obj \
    --type gen \
    --process_with_model_id