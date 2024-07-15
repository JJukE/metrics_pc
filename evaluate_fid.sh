export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/root/dev/scenegen/t2std/

# # NFD
# python scripts/TSD/metrics/evaluate_fid.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/render_for_fid_train_ids/03001627 \
#     --gen-dir /root/hdd1/T2STD/res/NFD/render_for_fid

# TSD
python scripts/TSD/metrics/evaluate_fid.py \
    --gt-dir /root/hdd2/ShapeNetCoreV2/render_for_fid_train_ids/03001627/ \
    --gen-dir /root/hdd1/t2std_a100/res/TSD_train_ids/240606_2333_generation
