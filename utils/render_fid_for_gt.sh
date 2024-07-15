export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=/root/dev/scenegen/t2std/
# python scripts/TSD/metrics/utils/render_fid.py \
#     --gen_dir /root/hdd2/ShapeNetCoreV2/no_mtl_objs/02691156 \
#     --gen_out_dir /root/hdd2/ShapeNetCoreV2/render_for_fid_no_light/02691156 \
#     --num_views 20 \
#     --suffix nomtl.obj \
#     --process_with_model_id

# python scripts/TSD/metrics/utils/render_fid.py \
#     --gen_dir /root/hdd2/ShapeNetCoreV2/no_mtl_objs/02958343 \
#     --gen_out_dir /root/hdd2/ShapeNetCoreV2/render_for_fid_no_light/02958343 \
#     --num_views 20 \
#     --suffix nomtl.obj \
#     --process_with_model_id

python scripts/TSD/metrics/utils/render_fid.py \
    --gen_dir /root/hdd2/ShapeNetCoreV2/no_mtl_objs/03001627 \
    --gen_out_dir /root/hdd2/ShapeNetCoreV2/render_for_fid_train_ids/03001627 \
    --num_views 20 \
    --suffix nomtl.obj \
    --split_path /root/dataset_sj/t2std/pseudo_gt/split \
    --cat_id "03001627" \
    --process_with_model_id

# python scripts/TSD/metrics/utils/render_fid.py \
#     --gen_dir /root/hdd2/ShapeNetCoreV2/no_mtl_objs/04379243 \
#     --gen_out_dir /root/hdd2/ShapeNetCoreV2/render_for_fid_no_light/04379243 \
#     --num_views 20 \
#     --suffix nomtl.obj \
#     --process_with_model_id 