
files_path= # path to the files
dataset_path= # path to the dataset

seed=0


######################### small #################################
scale="small"
datasets_scale='datacomp_small' #'cc12m'   #

# judge ocr
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/T_MARS_small --save_path ${files_path}/${datasets_scale}/uids/judge_ocr.npy --name judge_ocr \
#          --arch l14 --fraction 0.3 --save_dataset_path ${files_path}/${datasets_scale}/OCR/ --files_path ${files_path}

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/T_MARS_metadata3 --save_path ${files_path}/${datasets_scale}/uids/no_ocr.npy --name judge_ocr \
#          --arch l14 --fraction 0.3 --save_dataset_path ${dataset_path}/${datasets_scale}/OCR/ --files_path ${files_path}

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.44_fvas0.3_no_ocr.npy \
#         --files_path ${files_path} --name vas_v2 --arch l14 --fraction 0.44 --fraction_vas 0.3 --target_variance_name 'imagenet-1k' --given_uids_path ${files_path}/${datasets_scale}/uids/no_ocr.npy

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.4_fvas0.3_no_ocr.npy \
#         --files_path ${files_path} --name vas_v2 --arch l14 --fraction 0.4 --fraction_vas 0.3 --target_variance_name 'imagenet-1k' --given_uids_path ${files_path}/${datasets_scale}/uids/no_ocr.npy

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.5_fvas0.5_no_ocr.npy \
#         --files_path ${files_path} --name vas_v2 --arch l14 --fraction 0.5 --fraction_vas 0.5 --target_variance_name 'imagenet-1k' --given_uids_path ${files_path}/${datasets_scale}/uids/no_ocr.npy

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_t0.2142_fvas0.3_no_ocr.npy \
#         --files_path ${files_path} --name vas_v2 --arch l14 --threshold 0.2142 --fraction_vas 0.3 --target_variance_name 'imagenet-1k' --given_uids_path ${files_path}/${datasets_scale}/uids/no_ocr.npy

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.5_fvas0.3.npy \
#         --files_path ${files_path} --name vas --arch l14 --fraction 0.5 --fraction_vas 0.3 --target_variance_name 'imagenet-1k'

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.4_fvas0.3.npy \
#         --files_path ${files_path} --name vas --arch l14 --fraction 0.4 --fraction_vas 0.3 --target_variance_name 'imagenet-1k'


# vas_v3

# soft_type_list=(
#         # 1.0
#         # 2.0
#         # 3.0
#         # 5.0
#         # 7.0
#         # 10.0
#         # 20.0
#         # -5.0
#         # -10.0
#         # -20.0
#         # -50.0
# )

# for soft_type in "${soft_type_list[@]}"
# do
#         echo "soft_type = ${soft_type}"
#         python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.44_fvas0.3_v3_${soft_type}.npy \
#                 --files_path ${files_path} --name vas_v3 --arch l14 --fraction 0.44 --fraction_vas 0.3 --target_variance_name 'imagenet_1k' --soft_type ${soft_type}

# done

# VAS curve
# vass min, max, mean = 0.1289430558681488, 0.3679245114326477, 0.2370605170726776
# css min, max, mean = 0.1600341796875, 0.369873046875, 0.25037717819213867
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_t0.16_th0.37_tv0.08_ce1.0_f0.3.npy \
                # --files_path ${files_path} --name vas_curve --arch l14 --fraction 0.3 --threshold 0.16 --threshold_high 0.37 --threshold_vas 0.08 --clipscore_exponent 1.0 --target_variance_name 'imagenet_1k'


# vass min, max, mean = 0.1289430558681488, 0.3679245114326477, 0.23706036806106567
# css min, max, mean = 0.1600341796875, 0.369873046875, 0.25037702918052673
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${
#         datasets_scale}/uids/vas_t0.16_th0.37_tv0.1_ce1.0_f0.3.npy \
#                 --files_path ${files_path} --name vas_curve --arch l14 --fraction 0.3 --threshold 0.16 --threshold_high 0.37 --threshold_vas 0.1 --clipscore_exponent 1.0 --target_variance_name 'imagenet_1k'



# vass min, max, mean = 0.13560821115970612, 0.3679245114326477, 0.23952433466911316
# css min, max, mean = 0.1600341796875, 0.369873046875, 0.24750548601150513
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_t0.16_th0.37_tv0.08_ce0.9_f0.3.npy \
#                 --files_path ${files_path} --name vas_curve --arch l14 --fraction 0.3 --threshold 0.16 --threshold_high 0.37 --threshold_vas 0.08 --clipscore_exponent 0.9 --target_variance_name 'imagenet_1k'


# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_t0.19_th0.37_tv0.08_ce1.0_f0.3.npy \
#                 --files_path ${files_path} --name vas_curve --arch l14 --fraction 0.3 --threshold 0.19 --threshold_high 0.37 --threshold_vas 0.08 --clipscore_exponent 1.0 --target_variance_name 'imagenet_1k'


# VAS-D(Traindata) + clip score 45%, first clip score then vas, reproduce 
# num_iters=168
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_f0.44_fvas0.3_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch l14  --fraction 0.44 --fraction_vas 0.3  --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# num_iters=168
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.8_t0.214233_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.8 --threshold 0.214233 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# num_iters=168
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.6_t0.214233_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.6 --threshold 0.214233 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# num_iters=500
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.7_f0.3_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.7 --fraction 0.3 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000


# num_iters=168
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.7_t0.214233_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.7 --threshold 0.214233 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# order test
# num_iters=168
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.75_t0.214233_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.75 --threshold 0.214233 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# # # # VAS-D(Traindata) + clip score 45% b32
# num_iters=500
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_b32_f0.44_fvas0.3_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch b32 --fraction 0.44 --fraction_vas 0.3 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# b32
# image + clip score
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/image_based_b32_intersect_clip_score_l14_44_percent.npy --name image_based_intersect_clip_score --image_based_scale small --batch_size 512 --arch b32 --fraction 0.44 \
#                 --cache_path ${files_path}/cache --files_path ${files_path}

# # image
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/image_based_b32.npy --name image_based --image_based_scale small --batch_size 512 --arch b32 \
#                 --cache_path ${files_path}/cache --files_path ${files_path}
# num_iters=200
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_b32_f0.44_fvas0.3_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch b32 --fraction 0.44 --fraction_vas 0.3 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000


# num_iters=1
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.75_t0.214233_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.75 --threshold 0.214233 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000


# num_iters=1
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.7_t0.214233_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.7 --threshold 0.214233 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# num_iters=1
# # b32 VAS-D(Traindata) + clip score 45%, first clip score then vas
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_b32_f0.44_fvas0.3_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch b32 --fraction 0.44 --fraction_vas 0.3 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# num_iters=200

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_b32_iesame_f0.44_fvas0.3_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch b32 --fraction 0.44 --fraction_vas 0.3 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000 --update_image_feature_arch 0

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_b32_f0.44_fvas0.35_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch b32 --fraction 0.44 --fraction_vas 0.35 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_b32_f0.44_fvas0.27_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch b32 --fraction 0.44 --fraction_vas 0.27 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000
# VAS-D(Traindata) + clip score 45%, first vas then clip score


# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.85_f0.3_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.85 --fraction 0.3 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.8_f0.3_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.8 --fraction 0.3 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_v2_fvas0.6_f0.3_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d_v2 --arch l14  --fraction_vas 0.6 --fraction 0.3 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 5000000 --batch_size_vass 1000000

target_variance_name_list=(
        # interpolated_in1k_eurosat_svhn_kitti_0.9_0.04_0.03_0.03
        # interpolated_in1k_eurosat_svhn_kitti_0.94_0.02_0.02_0.02
        # interpolated_in1k_eurosat_svhn_0.91_0.05_0.04
        # interpolated_in1k_eurosat_svhn_kitti_0.92_0.04_0.02_0.02
        # vas_d_fvas0.3_200
)
for target_variance_name in "${target_variance_name_list[@]}"
do
        echo "target_variance_name = ${target_variance_name}"
        python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_v2_f0.44_fvas0.3_${target_variance_name}.npy \
                --files_path ${files_path} --name vas_v2 --arch l14 --fraction 0.44 --fraction_vas 0.3 --target_variance_name ${target_variance_name}
done

###################### b32 test #################################
# clip score
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_b32_30_percent.npy --name clip_score --arch b32 --fraction 0.3

# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_b32_44_percent.npy --name clip_score --arch b32 --fraction 0.44

# VAS(ImageNet-1k) v2 + clip score 45%
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_b32_f0.44_fvas0.3.npy \
#         --files_path ${files_path} --name vas --arch b32 --fraction 0.44 --fraction_vas 0.3 --target_variance_name 'imagenet-1k'

# vas variant
# echo "vas variant for small scale"

# for norm in 1 3 5
# do 
#         echo "norm = ${norm}"
#         time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_variant_f0.44_fvas0.3_${norm}.npy \
#                 --files_path ${files_path} --name vas_variant --arch l14 --fraction 0.44 --fraction_vas 0.3  --cache_path ${files_path}/cache --batch_size 5000 --norm ${norm}
# done

# val proxy
# for norm in 100 20 30 1 3 5
# do 
#         echo "norm = ${norm}"
#         time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_variant_val_f0.44_fvas0.3_${norm}.npy \
#                 --files_path ${files_path} --name vas_variant --arch l14 --fraction 0.44 --fraction_vas 0.3  --cache_path ${files_path}/cache --batch_size 8000 --norm ${norm} --proxy_name 'validation' --proxy_path ${files_path}/variance/tmp_features/
# done



# new clip score
# fraction=0.3
# score_type=0
# temperature=0.2
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}

# fraction=0.3
# score_type=0
# temperature=1.0
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}

# fraction=0.3
# score_type=0
# temperature=5.0
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}

# fraction=0.3
# score_type=0
# temperature=0.005
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}

# fraction=0.44
# score_type=0
# temperature=0.01
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}

# fraction=0.6
# score_type=0
# temperature=0.01
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}
# vas + new clip score
# fraction=0.44
# fraction_vas=0.3006
# score_type=0
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k'



# add average
# fraction=0.3
# fraction_vas=0.3
# score_type=0
# average_num=3
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}

# fraction=0.3
# fraction_vas=0.3
# score_type=0
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}


# fraction=0.45
# fraction_vas=0.3006
# score_type=0
# average_num=5
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}

# fraction=0.2
# score_type=0
# temperature=0.01
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}

# fraction=0.8
# score_type=0
# temperature=0.01
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}

# fraction=0.1
# score_type=0
# temperature=0.01
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}


# vas + new clip score
# fraction=0.46
# fraction_vas=0.3006
# score_type=0
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k'


# fraction=0.44
# fraction_vas=0.3006
# score_type=0
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k'

# fraction=0.43
# fraction_vas=0.3006
# score_type=0
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k'

# fraction=0.44
# threshold_vas=0.153
# score_type=0
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_tvas${threshold_vas}_st${score_type}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --threshold_vas ${threshold_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k'

# fraction=0.44
# fraction_vas=0.31
# score_type=0
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k'

######################### medium #################################
scale="medium"
datasets_scale='datacomp_medium' #'cc12m'   #




######################## b32 #################################
# clip score

# clip score
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_b32_30_percent.npy --name clip_score --arch b32 --fraction 0.3

# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_b32_20_percent.npy --name clip_score --arch b32 --fraction 0.2

# # VAS(ImageNet-1k) v2 + clip score 30%
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_b32_f0.3_fvas0.2.npy \
#         --files_path ${files_path} --name vas --arch b32 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'imagenet-1k'

# # VAS-D(Traindata) + clip score 30%
# num_iters=168
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_b32_f0.3_fvas0.2_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch b32 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 500000 --batch_size_vass 200000
        
# medium baselines

# clip score
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_l14_20_percent.npy --name clip_score --arch l14 --fraction 0.2
# echo "clip score"

# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_l14_10_percent.npy --name clip_score --arch l14 --fraction 0.1
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_l14_40_percent.npy --name clip_score --arch l14 --fraction 0.4
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_l14_50_percent.npy --name clip_score --arch l14 --fraction 0.5
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_l14_75_percent.npy --name clip_score --arch l14 --fraction 0.75
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_l14_90_percent.npy --name clip_score --arch l14 --fraction 0.9


# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_l14_5_percent.npy --name clip_score --arch l14 --fraction 0.05
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/clip_score_l14_1_percent.npy --name clip_score --arch l14 --fraction 0.01
# image + clip score
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/image_based_intersect_clip_score_l14_30_percent.npy --name image_based_intersect_clip_score --image_based_scale medium --batch_size 512 --arch l14 --fraction 0.3 \
#                 --cache_path ${files_path}/cache --files_path ${files_path}

# # image
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/image_based.npy --name image_based --image_based_scale medium --batch_size 512 \
#                 --cache_path ${files_path}/cache --files_path ${files_path}

# text
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/text_based.npy --name text_based \
#                 --cache_path ${files_path}/cache --files_path ${files_path}

# text
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/text_based_test.npy --name text_based \
#                 --cache_path ${files_path}/cache --files_path ${files_path}

# basic
# # basic
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/basic_filter.npy --name basic_filter \
#                 --cache_path ${files_path}/cache --files_path ${files_path}

# # laion2b
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/laion2b.npy --name laion2b \
#                 --cache_path ${files_path}/cache --files_path ${files_path}

# vas_v2

# time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.3_fvas0.2_tmp.npy \
#         --files_path ${files_path} --name vas --arch l14 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'imagenet-1k'

# time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.3_fvas0.2_v2.npy \
#         --files_path ${files_path} --name vas_v2 --arch l14 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'imagenet-1k'

# vas variant
# echo "vas variant"
# for norm in 10 3
# do      
#         echo "norm = ${norm}"
#         time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_variant_f0.3_fvas0.2_${norm}.npy \
#                 --files_path ${files_path} --name vas_variant --arch l14 --fraction 0.3 --fraction_vas 0.2  --cache_path ${files_path}/cache --batch_size 5000 --norm ${norm}
# done

# for norm in 10 30 5
# do      
#         echo "norm = ${norm}"
#         time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_variant_val_f0.3_fvas0.2_${norm}.npy \
#                 --files_path ${files_path} --name vas_variant --arch l14 --fraction 0.3 --fraction_vas 0.2  --cache_path ${files_path}/cache --batch_size 8000 --norm ${norm} --proxy_name 'validation' --proxy_path ${files_path}/variance/tmp_features/
# done

# for norm in 100 20
# do      
#         echo "norm = ${norm}"
#         time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_variant_img1k_val_f0.3_fvas0.2_${norm}.npy \
#                 --files_path ${files_path} --name vas_variant --arch l14 --fraction 0.3 --fraction_vas 0.2  --cache_path ${files_path}/cache --batch_size 4000 --norm ${norm} --proxy_name 'imagenet_1k_validation' --proxy_path ${files_path}/variance/tmp_features/
# done

# new clip score
# fraction=0.3
# score_type=0
# temperature=0.01
# batch_size=32768
# echo "new clip score with fraction = ${fraction}, score_type = ${score_type}, temperature = ${temperature}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_st${score_type}_tem${temperature}_bs${batch_size}.npy \
#         --name cs_new --fraction ${fraction} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --temperature ${temperature}

# vas variant + new clip score
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=100
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_val_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/variance/tmp_features/ --batch_size_vass 6000 --cache_path ${files_path}/cache --norm ${norm}

# fraction=0.2
# fraction_vas=0.1
# score_type=0
# average_num=20
# norm=100
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_val_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/variance/tmp_features/ --batch_size_vass 6000 --cache_path ${files_path}/cache --norm ${norm} # 95 minutes

# fraction=0.275
# fraction_vas=0.175
# score_type=0
# average_num=20
# norm=100
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_val_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/variance/tmp_features/ --batch_size_vass 6000 --cache_path ${files_path}/cache --norm ${norm}

# # # CLIPLoss \cap VAS_inf(Target)
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=101
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_val_train_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train/train_features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm}

############################# VAS 2 #############################
# # CLIPLoss (l14) \cap VAS_inf(Target) (l14)
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=101
# arch=l14
# arch_ncl=l14
# metadata_dir_name=metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train/train_features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=0
# norm=101
# arch=b32
# arch_ncl=b32
# metadata_dir_name=metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# vas_inf_type=99
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 10000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}


# # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# # sleep 2h
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}


# CLIPLoss (b32) \cap VAS_inf(Target) (l14)
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=101
# arch=l14
# arch_ncl=b32
# metadata_dir_name=metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train/train_features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# # CLIPLoss (b32) 30%
# fraction=0.3
# fraction_vas=0.3
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=b32
# metadata_dir_name=metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# new_str=new
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, new_str = ${new_str}, temperature = ${temperature}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_${new_str}_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# # CLIPScore (b32)30%
# fraction=0.3
# fraction_vas=0.3
# score_type=0
# average_num=0
# norm=100
# arch=b32
# arch_ncl=b32
# metadata_dir_name=metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# if_use_old_cs=1
# new_str=old
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_${new_str}_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --if_use_old_cs ${if_use_old_cs}

# # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32) tem 0.01
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}


# CLIPLoss (dfn_p) \cap VAS_inf(Target) (l14) tem 0.01
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=100
# arch=l14
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}



# CLIPLoss (b32) \cap VAS_inf(Target) (b32) tem 0.01
fraction=0.125
fraction_vas=0.1
score_type=0
average_num=10
norm=100
arch=b32
arch_ncl=b32
metadata_dir_name=metadata
# proxy_f_name=eval_train/train_features
proxy_name=validation
proxy_f_dir_name=eval_train_b32/features
temperature=0.01
echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
        --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
        --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# # CLIPLoss (b32) \cap VAS_2(IN-1k) (b32) tem 0.01
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=2
# arch=b32
# arch_ncl=b32
# metadata_dir_name=metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation # always validation
# proxy_f_dir_name=eval_train_b32/in1k_features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}


# CLIPScore (b32) 20%
# fraction=0.2
# fraction_vas=0.2
# score_type=0
# average_num=0
# norm=100
# arch=b32
# arch_ncl=b32
# metadata_dir_name=metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# if_use_old_cs=1
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --if_use_old_cs ${if_use_old_cs}


# # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# sleep 2h
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}


# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=101
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}



# num_iters=200
# echo 'run vas_d with num_iters = 200'
# time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_f0.3_fvas0.2_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch l14 --fraction 1.0 --fraction_vas 0.75 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 1000000 --batch_size_vass 500000


################ some wait to run ################


# # # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.2
# fraction_vas=0.175
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# # # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.175
# fraction_vas=0.15
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# # # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.2
# fraction_vas=0.15
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}


######################################### VISUALIZATION #########################################


# # # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.15
# fraction_vas=0.1
# score_type=0
# average_num=2
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

############################## final war #################################


# # given uids
# # given_uids_path=${files_path}/${datasets_scale}/uids/datacomp_medium_dfn_20m_inds.npy
# outside_uids_name=datacomp_medium_dfn_20m_inds
# given_uids_path=${files_path}/${datasets_scale}/uids/${outside_uids_name}.npy
# # # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.9
# fraction_vas=0.8
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# vas_inf_type=0
# echo "givenuids ${given_uids_path}"
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}_${outside_uids_name}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type} \
#         --given_uids_path $given_uids_path



# # given uids
# # given_uids_path=${files_path}/${datasets_scale}/uids/datacomp_medium_hype_115622300_10p.npy
# outside_uids_name=datacomp_medium_hype_115622300_10p
# given_uids_path=${files_path}/${datasets_scale}/uids/${outside_uids_name}.npy
# # # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.9
# fraction_vas=0.7
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# vas_inf_type=0
# echo "givenuids ${given_uids_path}"
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}_${outside_uids_name}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type} \
#         --given_uids_path $given_uids_path

        
# #### no cliploss
# # given uids
# # given_uids_path=${files_path}/${datasets_scale}/uids/datacomp_medium_dfn_20m_inds.npy
# outside_uids_name=datacomp_medium_dfn_20m_inds
# given_uids_path=${files_path}/${datasets_scale}/uids/${outside_uids_name}.npy
# # # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=1.0
# fraction_vas=0.9
# average_num=0
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# vas_inf_type=0
# echo "givenuids ${given_uids_path}"
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}_${outside_uids_name}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type} \
#         --given_uids_path $given_uids_path



# # given uids
# # given_uids_path=${files_path}/${datasets_scale}/uids/datacomp_medium_dfn_20m_inds.npy
# outside_uids_name=datacomp_medium_dfn_20m_inds
# given_uids_path=${files_path}/${datasets_scale}/uids/${outside_uids_name}.npy
# # # CLIPLoss (b32) \cap VAS_inf(Target) (b32)
# fraction=0.9
# fraction_vas=0.8
# average_num=10
# norm=100
# arch=b32
# arch_ncl=b32
# metadata_dir_name=metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# vas_inf_type=0
# echo "givenuids ${given_uids_path}"
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}_${outside_uids_name}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type} \
#         --given_uids_path $given_uids_path



# # given uids
# # given_uids_path=${files_path}/${datasets_scale}/uids/datacomp_medium_hype_115622300_10p.npy
# outside_uids_name=datacomp_medium_hype_115622300_10p
# given_uids_path=${files_path}/${datasets_scale}/uids/${outside_uids_name}.npy
# # # CLIPLoss (b32) \cap VAS_inf(Target) (b32)
# fraction=0.9
# fraction_vas=0.7
# average_num=10
# norm=100
# arch=b32
# arch_ncl=b32
# metadata_dir_name=metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# vas_inf_type=0
# echo "givenuids ${given_uids_path}"
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}_${outside_uids_name}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type} \
#         --given_uids_path $given_uids_path


# given uids
# given_uids_path=${files_path}/${datasets_scale}/uids/datacomp_medium_hype_115622300_10p.npy
# outside_uids_name=cs_new_b32_f0.2_target24_b32_fvas0.1_st_an10_norm100_tem0.01_intersect_cs_new_f0.2_val_fvas0.1_st0_an20_norm100
# outside_uids_name=cs_new_f0.3_val_train_fvas0.2_st0_an20_norm100_intersect_cs_new_b32_f0.3_val_train_fvas0.2_st0_an20_norm100
# given_uids_path=${files_path}/${datasets_scale}/uids/${outside_uids_name}.npy
# # # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.9
# fraction_vas=0.8
# average_num=2
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# vas_inf_type=0
# echo "givenuids ${given_uids_path}"
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}_${outside_uids_name}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type} \
#         --given_uids_path $given_uids_path

############################## final war end #################################

# # # CLIPLoss (dfn_p) standard
# fraction=0.175
# fraction_vas=0.175
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.07
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}_16384.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 16384 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.125
# fraction_vas=0.1
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# CLIPLoss (l14) \cap VAS_inf(Target) (l14)
# fraction=0.2
# fraction_vas=0.1
# average_num=10
# norm=100
# arch=l14
# arch_ncl=l14
# metadata_dir_name=metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train/new_features
# temperature=0.01
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}


# # CLIPLoss (l14) \cap VAS_inf(Target) (l14)
# fraction=0.2
# fraction_vas=0.1
# average_num=2
# norm=100
# arch=l14
# arch_ncl=l14
# metadata_dir_name=metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train/train_features
# temperature=0.01
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}



# CLIPLoss (l14) \cap VAS_inf(Target) (l14)
# fraction=0.2
# fraction_vas=0.1
# average_num=2
# norm=100
# arch=l14
# arch_ncl=l14
# metadata_dir_name=metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train/train_features
# temperature=0.01
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}

# CLIPLoss (dfn_p) \cap VAS_inf(Target) (dfn_p)
# fraction=0.2
# fraction_vas=0.1
# average_num=2
# norm=100
# arch=dfn_p
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_dfn_p/features
# temperature=0.01
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}

# # CLIPLoss (b32) \cap VAS_inf(Target) (l14)
# fraction=0.2
# fraction_vas=0.1
# average_num=2
# norm=100
# arch=l14
# arch_ncl=b32
# metadata_dir_name=metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train/train_features
# temperature=0.01
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}


# CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.2
# fraction_vas=0.15
# average_num=2
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}

# # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.175
# fraction_vas=0.15
# average_num=2
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.07
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}

###################################### VISUALIZATION END ######################################

# # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32) # wait
# fraction=0.175
# fraction_vas=0.15
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.07
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}

# # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32) # wait
# fraction=0.175
# fraction_vas=0.15
# average_num=10
# norm=2
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}

# # wait
# # # CLIPLoss (dfn_p) 15%
# fraction=0.15
# fraction_vas=0.15
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.07
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# fraction=0.15 # wait
# fraction_vas=0.15
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

# # CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32) # wait
# fraction=0.175
# fraction_vas=0.15
# average_num=10
# norm=2
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.07
# vas_inf_type=0
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type 0 --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature} --vas_inf_type ${vas_inf_type}


####################### WAIT Group END #######################


# # CLIPLoss (dfn_p) 
# fraction=0.125
# fraction_vas=0.125
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}


# CLIPLoss (dfn_p) \cap VAS_inf(Target) (b32)
# fraction=0.175
# fraction_vas=0.125
# score_type=0
# average_num=10
# norm=100
# arch=b32
# arch_ncl=dfn_p
# metadata_dir_name=dfn-p-b32-both-metadata
# # proxy_f_name=eval_train/train_features
# proxy_name=validation
# proxy_f_dir_name=eval_train_b32/features
# temperature=0.01
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, arch = ${arch}, arch_ncl = ${arch_ncl}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/${metadata_dir_name} --save_path ${files_path}/${datasets_scale}/uids/cs_new_${arch_ncl}_f${fraction}_target24_${arch}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch ${arch} --arch_ncl ${arch_ncl} --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name ${proxy_name} --proxy_path ${files_path}/${proxy_f_dir_name}/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --temperature ${temperature}

#############################################################################


# # DFN-p CLIPLoss 17.5%

# fraction=0.175
# fraction_vas=0.175
# score_type=0
# batch_size=32768
# average_num=10
# norm=100
# if_use_old_cs=0
# temperature=0.07
# echo "DFN-p new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}, if_use_old_cs = ${if_use_old_cs}, norm = ${norm}, temperature = ${temperature}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/dfn-p-metadata --save_path ${files_path}/${datasets_scale}/uids/dfn_p_cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_tem${temperature}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch dfn_p --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train_dfn_p/features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs ${if_use_old_cs} --temperature ${temperature}

# fraction=0.25
# fraction_vas=0.1
# score_type=0
# average_num=10
# norm=101
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_val_train_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train/train_features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm}

# # # CLIPLoss \cap VAS_inf(Target) add target
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=101
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_target26_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train/new_features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm}

# # CLIPLoss \cap inf-AS(Target) --> type 2
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=100
# vas_inf_type=2
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}, vas_inf_type = ${vas_inf_type}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_val_train_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}_type${vas_inf_type}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train/train_features/ --batch_size_vass 2500 --cache_path ${files_path}/cache --norm ${norm} --vas_inf_type ${vas_inf_type}

# # CLIPLoss \cap VAS_inf(Target)
# fraction=0.25
# fraction_vas=0.1
# score_type=0
# average_num=10
# norm=2 #100
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_val_train_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train/train_features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm}


# # VAS_2(Target)
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=2
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_val_train_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train/train_features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm}

# # VAS_inf(IN-1k)
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=10
# norm=100
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_in1k_train_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train/in1k_features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm}

# CLIP Score \cap VAS_2(Target)

# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=0
# norm=2
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_old_f${fraction}_target_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'target' --average_num ${average_num} \
#          --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs 1


# CLIP Score 20% using b32

# fraction=0.2
# fraction_vas=0.2
# score_type=0
# average_num=0
# norm=2
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_old_b32_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch b32 --batch_size 32768 --score_type ${score_type} --target_variance_name 'target_delete_mean' --average_num ${average_num} \
#          --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs 1



# # CLIP Score \cap VAS_inf(Target)
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=0
# norm=100
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_old_f${fraction}_target_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train/train_features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs 1

# vas + new clip score
# fraction=0.3
# fraction_vas=0.3
# score_type=0
# batch_size=32768
# average_num=5
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} 


# test different average num
# fraction=0.3
# fraction_vas=0.3
# score_type=0
# batch_size=32768
# average_num=5
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} #

# fraction=0.3
# fraction_vas=0.3
# score_type=0
# batch_size=32768
# average_num=50
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} #

# fraction=0.2
# fraction_vas=0.2
# score_type=0
# batch_size=32768
# average_num=5
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}

# fraction=0.2
# fraction_vas=0.2
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}

# for visuaization
# fraction=0.2
# fraction_vas=0.2
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}


# fraction=0.2
# fraction_vas=0.2
# score_type=0
# batch_size=32768
# average_num=20
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}
# 58min * 7 L40, no cal for VAS

# fraction=0.01
# fraction_vas=0.01
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --average_num ${average_num}
# # 29 min

# fraction=0.05
# fraction_vas=0.05
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --average_num ${average_num}

# fraction=0.9
# fraction_vas=0.9
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --average_num ${average_num}
# # 32 min


###################### DFN_P ######################
### old clip score 30%
# fraction=0.3
# fraction_vas=0.3
# score_type=0
# batch_size=32768
# average_num=0
# norm=100
# if_use_old_cs=1
# echo "DFN-p new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}, if_use_old_cs = ${if_use_old_cs}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/dfn-p-metadata --save_path ${files_path}/${datasets_scale}/uids/dfn_p_cs_old_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch dfn_p --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train_dfn_p/features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs ${if_use_old_cs}

### old clip score 20%
# fraction=0.2
# fraction_vas=0.2
# score_type=0
# batch_size=32768
# average_num=0
# norm=100
# if_use_old_cs=1
# echo "DFN-p new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}, if_use_old_cs = ${if_use_old_cs}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/dfn-p-metadata --save_path ${files_path}/${datasets_scale}/uids/dfn_p_cs_old_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch dfn_p --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train_dfn_p/features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs ${if_use_old_cs}

### new clip score 30%
# fraction=0.3
# fraction_vas=0.3
# score_type=0
# batch_size=32768
# average_num=10
# norm=100
# if_use_old_cs=0
# echo "DFN-p new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}, if_use_old_cs = ${if_use_old_cs}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/dfn-p-metadata --save_path ${files_path}/${datasets_scale}/uids/dfn_p_cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch dfn_p --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train_dfn_p/features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs ${if_use_old_cs}

### new clip score 20%
# fraction=0.2
# fraction_vas=0.2
# score_type=0
# batch_size=32768
# average_num=10
# norm=100
# if_use_old_cs=0
# echo "DFN-p new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}, if_use_old_cs = ${if_use_old_cs}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/dfn-p-metadata --save_path ${files_path}/${datasets_scale}/uids/dfn_p_cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch dfn_p --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train_dfn_p/features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs ${if_use_old_cs}

### new clip score 17.5%
# fraction=0.175
# fraction_vas=0.175
# score_type=0
# batch_size=32768
# average_num=10
# norm=100
# if_use_old_cs=0
# echo "DFN-p new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}, if_use_old_cs = ${if_use_old_cs}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/dfn-p-metadata --save_path ${files_path}/${datasets_scale}/uids/dfn_p_cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch dfn_p --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train_dfn_p/features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs ${if_use_old_cs}

# ### CLIP Loss 30% \cap VAS_inf(Target)
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# batch_size=32768
# average_num=10
# norm=100
# if_use_old_cs=0
# echo "DFN-p new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}, if_use_old_cs = ${if_use_old_cs}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/dfn-p-metadata --save_path ${files_path}/${datasets_scale}/uids/dfn_p_cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch dfn_p --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train_dfn_p/features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs ${if_use_old_cs}

###################### DFN_P end ######################

# # CLIPLoss \cap VAS_inf(Target)
# fraction=0.275
# fraction_vas=0.075
# score_type=0
# average_num=10
# norm=100
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_val_train_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train/train_features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm}
        
# fraction=0.3
# fraction_vas=0.3
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_b32_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch b32 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}

# fraction=0.2
# fraction_vas=0.2
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_b32_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch b32 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}

# # CLIPLoss 30% \cap VAS_inf(Target) using b32
# fraction=0.3
# fraction_vas=0.2
# score_type=0
# average_num=20
# norm=100
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, average_num = ${average_num}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_b32_f${fraction}_val_train_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch b32 --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train_b32/features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm}

### dfn_p old clip score 17.5%
# fraction=0.175
# fraction_vas=0.175
# score_type=0
# batch_size=32768
# average_num=0
# norm=100
# if_use_old_cs=1
# echo "DFN-p new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}, if_use_old_cs = ${if_use_old_cs}, norm = ${norm}"
# python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/dfn-p-metadata --save_path ${files_path}/${datasets_scale}/uids/dfn_p_cs_old_f${fraction}_fvas${fraction_vas}_st${score_type}_an${average_num}_norm${norm}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch dfn_p --batch_size 32768 --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num} \
#         --proxy_name 'validation' --proxy_path ${files_path}/eval_train_dfn_p/features/ --batch_size_vass 3000 --cache_path ${files_path}/cache --norm ${norm} --if_use_old_cs ${if_use_old_cs}

# ### dfn


# fraction=0.4
# fraction_vas=0.4
# score_type=0
# batch_size=32768
# average_num=50
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --average_num ${average_num}
# # 123min

# fraction=0.2
# fraction_vas=0.2
# score_type=0
# batch_size=32768
# average_num=100
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --target_variance_name 'imagenet_1k' --average_num ${average_num}
# 240min * 8 L40, no cal for VAS

# new clip score for an=10, all about 35 min for 8 l40 hours
# fraction=0.1
# fraction_vas=0.1
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type}  --average_num ${average_num}
# # 

# fraction=0.4
# fraction_vas=0.4
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type}  --average_num ${average_num}


# fraction=0.5
# fraction_vas=0.5
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --average_num ${average_num}


# fraction=0.9
# fraction_vas=0.9
# score_type=0
# batch_size=32768
# average_num=10
# echo "new clip score with fraction = ${fraction}, fraction_vas = ${fraction_vas}, score_type = ${score_type}, batch_size = ${batch_size}, average_num = ${average_num}"
# time python baselines.py --files_path ${files_path} --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/cs_new_f${fraction}_fvas${fraction_vas}_st${score_type}_bs${batch_size}_an${average_num}.npy \
#         --name cs_new --fraction ${fraction} --fraction_vas ${fraction_vas} --arch l14 --batch_size ${batch_size} --score_type ${score_type} --average_num ${average_num}

# vas_d
num_iters=500
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_f0.3_fvas0.2_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch l14 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 2000000 --batch_size_vass 1000000

# time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_f0.3_fvas0.2_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch l14 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 2000000 --batch_size_vass 1000000

# num_iters=100
# time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_f0.3_fvas0.2_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch l14 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 2000000 --batch_size_vass 1000000


# num_iters=168
# time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_f0.3_fvas0.2_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch l14 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 2000000 --batch_size_vass 1000000

# num_iters=200
# time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_d_f0.3_fvas0.2_${num_iters}.npy \
#         --files_path ${files_path} --name vas_d --arch l14 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'self' --num_iters ${num_iters} --batch_size 1000000 --batch_size_vass 500000
# 321m

# vas
# time python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.3_fvas0.2.npy \
#         --files_path ${files_path} --name vas --arch l14 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'imagenet-1k'

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_t0.26_tvas0.17.npy \
#         --files_path ${files_path} --name vas --arch l14 --threshold 0.26 --threshold_vas 0.17 --target_variance_name 'imagenet-1k'

# vas fraction
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.3_fvas0.22.npy \
#         --files_path ${files_path} --name vas --arch l14 --fraction 0.3 --fraction_vas 0.22 --target_variance_name 'imagenet-1k'

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --save_path ${files_path}/${datasets_scale}/uids/vas_f0.3_fvas0.2_2.npy \
#         --files_path ${files_path} --name vas --arch l14 --fraction 0.3 --fraction_vas 0.2 --target_variance_name 'imagenet-1k'



# vas + T-MARS (mius)
# given_uids_path='uid_filtered_vitb_0p281_sorted_mediumscale_tmars'

# for target_variance_name in "${target_variance_name_list[@]}"
# do
#         echo "target_variance_name = ${target_variance_name}, scale = ${scale}, given_uids_path = ${given_uids_path}"
#         python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --files_path ${files_path} --save_path ${files_path}/${datasets_scale}/uids/vas_tmars_t0.0_fvas0.17_${target_variance_name}.npy \
#                 --name vas --arch l14 --target_variance_name 'imagenet-1k' --threshold 0.0 --fraction_vas 0.17 --given_uids_path ${files_path}/${datasets_scale}/uids/${given_uids_path}.npy --target_variance_name ${target_variance_name}
# done
# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --files_path ${files_path} --save_path ${files_path}/${datasets_scale}/uids/vas_tmars_t0.0_fvas0.18.npy \ 
#         --name vas --arch l14 --target_variance_name 'imagenet-1k' --threshold 0.0 --fraction_vas 0.18 --given_uids_path ${files_path}/${datasets_scale}/uids/${given_uids_path}.npy # the fraction = 0.18, the threshold = 0.10709410905838013, mean = 0.20115280151367188, max = 0.3693997859954834

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --files_path ${files_path} --save_path ${files_path}/${datasets_scale}/uids/vas_tmars_t0.0_fvas0.16.npy \
#         --name vas --arch l14 --target_variance_name 'imagenet-1k' --threshold 0.0 --fraction_vas 0.16 --given_uids_path ${files_path}/${datasets_scale}/uids/${given_uids_path}.npy 
        # tAfter filtering, the fraction = 0.16, the threshold = 0.1387079954147339, mean = 0.21077121794223785, max = 0.3693997859954834. len = 20480000

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --files_path ${files_path} --save_path ${files_path}/${datasets_scale}/uids/vas_tmars_t0.12_tvas0.12.npy \
#         --name vas --arch l14 --target_variance_name 'imagenet-1k' --threshold 0.12 --threshold_vas 0.12 --given_uids_path ${files_path}/${datasets_scale}/uids/${given_uids_path}.npy # the fraction = 0.19776778125, the threshold = 0.12005615234375, the fraction = 0.172979515625, the threshold = 0.12000001966953278

# python baselines.py --metadata_dir ${dataset_path}/${datasets_scale}/metadata --files_path ${files_path} --save_path ${files_path}/${datasets_scale}/uids/vas_tmars_t0.24_tvas0.17.npy \
#         --name vas --arch l14 --target_variance_name 'imagenet-1k' --threshold 0.24 --threshold_vas 0.17 --given_uids_path ${files_path}/${datasets_scale}/uids/${given_uids_path}.npy --if_add_more 1


