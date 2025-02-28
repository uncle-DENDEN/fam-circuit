#!/bin/bash
cwd="/user_data/weifanw/familiarity_plot"  # project directory

# =====================================================================================================
# set all parameter the same as the those used in the simulation
postfix="mix50%_all"
tau_train=150
gamma=30
tau_test_1=1000
si_1=50
tau_test_2=500
si_2=250

mkdir -p ${cwd}/log_assoc

# plot Fig4 and FigS3
# logfile=${cwd}/log_assoc/plot_assoc_Fig4,S3.txt

# input_path="input_famassoc/r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy"
# pre_path="response_famassoc/r_pre_${tau_test_1}ms_5_4x10_${si_1}_64_8_8_${postfix}.npy"
# post_path="response_famassoc/r_epoch0_tx${gamma}_${tau_test_1}ms_5_4x10_${si_1}_64_8_8_${postfix}.npy"
# post_path_hr="response_famassoc/r_epoch4_tx${gamma}_${tau_test_2}ms_5_4x10_${si_2}_64_8_8_${postfix}.npy"

# echo "plotting Fig4, FigS3..."
# srun -p cpu -n1 --mem=20GB --time=1:00:00 --pty bash << EOF &> $logfile &
#     module load anaconda3
#     source activate fam
#     cd ${cwd}
#     python Fig4,S3.py --input_path ${input_path} --pre_path ${pre_path} --post_path=${post_path} --post_path_hr ${post_path_hr}
# EOF
# pid0=$!
# wait $pid0

# plot Fig5 and FigS4
logfile=${cwd}/log_assoc/plot_assoc_Fig5,S4.txt

input_path="input_famassoc/r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy"
pre_response_path="response_famassoc/r_pre_${tau_test_1}ms_5_4x10_${si_1}_64_8_8_${postfix}.npy"
post_response_path="response_famassoc/r_epoch0_tx${gamma}_${tau_test_1}ms_5_4x10_${si_1}_64_8_8_${postfix}.npy"
jac_path="example_jac_matrix/jac_epoch0_img0_10%_sample0.pkl"
proj_dist_path="proj_dists_slow_1-200/sample_set0"

echo "plotting Fig5, FigS4..."
srun -p cpu -n1 --mem=80GB --time=1:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python Fig5,S4.py --input_path ${input_path} --pre_response_path ${pre_response_path} --post_response_path=${post_response_path} --jac_path=${jac_path} --proj_dist_path ${proj_dist_path}
EOF
pid1=$!
wait $pid1

