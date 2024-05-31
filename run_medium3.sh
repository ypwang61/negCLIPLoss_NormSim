################################################## datacomp_small#################################################
num_gpus=8
scale="medium"
seed=0 #42

datasets_scale='datacomp_medium' #'cc12m'   #

files_path= #path to the files folder

dataset_path= #path to the dataset folder

# define a list of the names of filters, for filter in a given list, echo the name of the filter
ulist=(
    # uid_filtered_vitb_0p281_sorted_mediumscale_tmars
    # vas_variant_f0.3_fvas0.2_10
    # vas_variant_f0.3_fvas0.2_3

    # cs_new_f0.3_val_fvas0.2_st0_an1_norm100
    # cs_new_f0.3_fvas0.3_st0_bs32768_an5
    # # cs_new_f0.2_fvas0.2_st0_bs32768_an10
    # cs_new_f0.3_fvas0.3_st0_bs32768_an50
    # cs_new_f0.2_fvas0.2_st0_bs32768_an50
    # cs_new_f0.2_fvas0.2_st0_bs32768_an5

    # clip_score_l14_90_percent
    # clip_score_l14_50_percent
    # clip_score_l14_40_percent
    # clip_score_l14_10_percent

    # clip_score_l14_1_percent
    
    # datacomp_medium_dfn_20m_inds
    # datacomp_medium_dfn_20m_inds
    # cs_new_f0.2_val_fvas0.1_st0_an20_norm100.npy_merge_datacomp_medium_dfn_20m_inds
    
    # cs_new_f0.3_val_train_fvas0.2_st0_an20_norm100
    # cs_new_f0.2_fvas0.2_st0_bs32768_an100
    # cs_new_f0.2_fvas0.2_st0_bs32768_an20 # finish
    # clip_score_l14_5_percent # finish
    # clip_score_l14_75_percent
    
    # clip_score_l14_1_percent
    # cs_new_f0.1_fvas0.1_st0_bs32768_an10

    # final_uids # 34.5
    # cs_new_f0.3_val_train_fvas0.2_st0_an10_norm2

    # cs_new_f0.3_in1k_train_fvas0.2_st0_an10_norm100
    # cs_new_f0.4_fvas0.4_st0_bs32768_an10

    # cs_old_f0.3_target_fvas0.2_st0_an0_norm100

    # cs_old_f0.3_target_fvas0.2_st0_an0_norm2
    # cs_new_f0.5_fvas0.5_st0_bs32768_an10

    # dfn_p_cs_old_f0.3_fvas0.3_st0_an0_norm100
    # dfn_p_cs_new_f0.3_fvas0.3_st0_an10_norm100

    # dfn_p_cs_new_f0.3_target_fvas0.2_st0_an10_norm100 # 32.5, 29.4, 23.6, 33.5, 24.2

    # cs_new_f0.3_val_train_fvas0.2_st0_an10_norm101_sum_rank
    # cs_new_b32_f0.3_val_train_fvas0.2_st0_an20_norm100
    # merge_f0.2_fvas0.1_dfn

    # cs_new_f0.3_val_train_fvas0.2_st0_an10_norm100_type2 # 34.0

    # cs_new_b32_f0.2_fvas0.2_st0_bs32768_an10
    # cs_new_f0.3_val_train_fvas0.1_st0_an10_norm100_merge_datacomp_medium_dfn_20m_inds
    
    # dfn_p_cs_new_f0.175_fvas0.175_st0_an10_norm100_tem0.07

    # dfn_p_cs_old_f0.175_fvas0.175_st0_an0_norm100
    # dfn_p_cs_new_f0.175_fvas0.175_st0_an10_norm100

    # cs_old_b32_f0.2_fvas0.2_st0_an0_norm2 # 32.2
    # cs_new_l14_f0.2_fvas0.1

    # cs_new_b32_f0.3_target24_l14_fvas0.2_st0_an10_norm101_tem0.01 # 35.1, interesting
    # merge_b32_f0.2_b32_fvas0.1_dfn
    # cs_new_dfn_p_f0.3_target24_b32_fvas0.2_st0_an10_norm100_tem0.01

    # cs_old_b32_f0.3_target24_b32_fvas0.3_st0_an0_norm100_tem0.01 # 0.333
    # cs_new_b32_f0.275_target24_b32_fvas0.175_st0_an10_norm100_tem0.01 # 0.343


    # merge_hype_dfn
    # cs_new_dfn_p_f0.2_target24_b32_fvas0.175_st0_an10_norm100_tem0.01

    # merge3
    # merge_hype_dfn_all_combine_db_97_98
    # merge_hype_db_97_dfn_db_98

    # merge_hype_dfn_db_0.125_0.1
    # cs_new_b32_f0.3_target24_b32_fvas0.3_st0_an10_norm100_tem0.01
    
    # merge5
    # hype
    # cs_new_f0.3_val_train_fvas0.2_st0_an20_norm100_intersect_cs_new_b32_f0.3_val_train_fvas0.2_st0_an20_norm100
    # merge_bbll3232

    # merge4
    cs_new_f0.3_fvas0.3_st0_bs32768_an50_intersect_vas_d_f0.3_fvas0.2_500

    # dfn_p_cs_new_f0.2_fvas0.2_st0_an10_norm100
    
    # dfn_p_cs_old_f0.2_fvas0.2_st0_an0_norm100
    # cs_new_f0.01_fvas0.01_st0_bs32768_an10
)

num_checkpoints=5

# sleep 1.5h
### get data shards
for filter in "${ulist[@]}"
do      
        if [ $filter == 'no_filter' ]
        then
            continue
        else
            # sharder
            echo "resharder begin for ${filter}"
            mkdir ${dataset_path}/${datasets_scale}/${filter}
            python resharder.py -i ${dataset_path}/${datasets_scale}/shards -o ${dataset_path}/${datasets_scale}/${filter} -s ${files_path}/${datasets_scale}/uids/${filter}.npy #--overwrite
            echo "resharder done for ${filter}"
        fi
done

# training 
for seed in ${seed}
do
    for filter in "${ulist[@]}" 
    do  
        exp_name="${filter}_${scale}_seed_${seed}"
        # exp_name=$filter

        if [ $filter == 'no_filter' ]
        then
            data_dir="${dataset_path}/${datasets_scale}/shards"
        else
            data_dir="${dataset_path}/${datasets_scale}/${filter}"
        fi

        # if num_checkpoints is not 5, add --num_checkpoints to exp_name
        if [ $num_checkpoints -ne 5 ]
        then
            exp_name="${exp_name}_ckpt${num_checkpoints}"
        fi

        # run 
        echo "training begin for ${exp_name}, data_dir = ${data_dir}"

        torchrun --rdzv_backend c10d --rdzv_endpoint localhost:29493 --nproc_per_node $num_gpus \
                train.py --scale $scale --data_dir $data_dir --output_dir ${files_path}/${datasets_scale}/output/ --exp_name ${exp_name} --num_checkpoints $num_checkpoints

        echo "training done for ${exp_name}, data_dir = ${data_dir}"
        # sleep 1.5h
    done
done

# evaluation
for seed in ${seed}
do
    for filter in "${ulist[@]}"
    do
        exp_name="${filter}_${scale}_seed_${seed}"
        # exp_name=$filter
        # if num_checkpoints is not 5, add --num_checkpoints to exp_name
        if [ $num_checkpoints -ne 5 ]
        then
            exp_name="${exp_name}_ckpt${num_checkpoints}"
        fi
        
        echo "evaluation begin for ${exp_name}"
        python evaluate.py  --train_output_dir ${files_path}/${datasets_scale}/output/${exp_name}/ --data_dir ${dataset_path}/datacomp_eval/
        echo "evaluation done for ${exp_name}"
    done
done