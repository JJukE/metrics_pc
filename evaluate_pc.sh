export CUDA_VISIBLE_DEVICES=1
PYTHONPATH=/root/dev/scenegen/t2std

# NFD
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/nfd/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# # NFD postprocessed
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/nfd_postprocessed_8000/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# # T2STD
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/T2STD/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# # T2STD postprocessed
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/T2STD_postprocessed/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# # TSD
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/TSD/240601_0158_generation/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# # TSD postprocessed
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/TSD_postprocessed/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# # TripLoRA
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/TripLoRA/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# # TripLoRA postprocessed
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/TripLoRA_postprocessed/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# # loras
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/pefts/lora/lora_main_128_400000_postprocessed/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# # lohas
# python evaluate_pc.py \
#     --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
#     --gen-dir /root/dataset_sj/t2std_a100/res/pefts/loha/loha_main_128_400000_postprocessed/pts_2048 \
#     --batch-size 512 \
#     --gt-interval 6778

# lokrs
python evaluate_pc.py \
    --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
    --gen-dir /root/dataset_sj/t2std_a100/res/pefts/lokr/lokr_main_1024_4_400000_postprocessed/pts_2048 \
    --batch-size 512 \
    --gt-interval 6778

python evaluate_pc.py \
    --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
    --gen-dir /root/dataset_sj/t2std_a100/res/pefts/lokr/lokr_realall_256_2_400000_postprocessed/pts_2048 \
    --batch-size 512 \
    --gt-interval 6778

python evaluate_pc.py \
    --gt-dir /root/hdd2/ShapeNetCoreV2/no_mtl_pc2048/03001627 \
    --gen-dir /root/dataset_sj/t2std_a100/res/pefts/lokr/lokr_realall_256_4_400000_postprocessed/pts_2048 \
    --batch-size 512 \
    --gt-interval 6778
