#!/bin/bash
cifar_path="/user_data/weifanw/cifar-100-python"  # cifar saving path
cwd="/user_data/weifanw/familiarity_clean"  # project directory

# =====================================================================================================
# common params
r_in_path="input_famassoc/r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy"
postfix="v5_wie=20_mix50%_all"
task="assoc"
wie=20

tau_train=150
gamma=30
num_epoch=5
out_path_train="weights_famassoc"

tau_test_1=1000
si_1=50
tau_test_2=500
si_2=250
out_path_test="response_famassoc"

mkdir -p ${cwd}/log_assoc

# get input
check_file="input_famassoc/r_in_cifar_all_noise_abs_5_4_10_64_8_8.npy"
if [ ! -f "$check_file" ]; then
  logfile=${cwd}/log_assoc/input_gen.txt
  echo "Generating feedforward inputs ... "
  srun -p cpu -n1 --mem=10GB --time=1:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python input_cifar.py --cifar_path ${cifar_path} --task ${task}
EOF
  pid0=$!
  wait $pid0

else
  echo "Input file existed, skipped."
fi

## get pre-trained response and BCM threshold
pids_1=()
swrp=0
wrp_path='.'

# pre-trained test1
check_file="response_famassoc/r_pre_${tau_test_1}ms_5_4x10_${si_1}_64_8_8_${postfix}.npy"
if [ ! -f "$check_file" ]; then
  BCM_thresh=1
  logfile=${cwd}/log_assoc/test_pre-trained1.txt
  echo "Running pre-trained testing1 ... "
  srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python test.py --r_in_path ${r_in_path} --out_path ${out_path_test} --task ${task} --tau_stim ${tau_test_1} --save_every_stim ${si_1} --wie ${wie} --set_wrp ${swrp} --wrp_path ${wrp_path} --postfix ${postfix} --get_BCM_threshold ${BCM_thresh}
EOF
  pids_1+=($!)

else
  echo "Output file exists for pre-trained testing1, skipped."
fi

# pre-trained test2
check_file="response_famassoc/r_pre_${tau_test_2}ms_5_4x10_${si_2}_64_8_8_${postfix}.npy"
if [ ! -f "$check_file" ]; then
  BCM_thresh=0
  logfile=${cwd}/log_assoc/test_pre-trained2.txt
  echo "Running pre-trained testing2 ... "
  srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python test.py --r_in_path ${r_in_path} --out_path ${out_path_test} --task ${task} --tau_stim ${tau_test_2} --save_every_stim ${si_2} --wie ${wie} --set_wrp ${swrp} --wrp_path ${wrp_path} --postfix ${postfix} --get_BCM_threshold ${BCM_thresh}
EOF
  pids_1+=($!)

else
  echo "Output file exists for pre-trained testing2, skipped."
fi

for pid in ${pids_1[@]}; do
    wait $pid
done

## run training
check_file="weights_famassoc/weight_epoch4_tx${gamma}_${tau_train}ms_5_4x10_64_8_8_${postfix}.npy"
if [ ! -f "$check_file" ]; then
  theta_path="input_famassoc/BCM_theta_cifar_all_noise_abs_5_4_10_64_8_8_ys_${postfix}.npy"
  logfile=${cwd}/log_assoc/train.txt
  echo "Running training ... "
  srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python train.py --r_in_path ${r_in_path} --theta_path ${theta_path} --out_path ${out_path_train} --task ${task} --gamma ${gamma} --tau_stim ${tau_train} --wie ${wie} --epoch ${num_epoch} --postfix ${postfix}
EOF
  pid2=$!
  wait $pid2

else
  echo "Output file exists for training, skipped."
fi

## run testing
pids_3=()
swrp=1
BCM_thresh=0

# test1
for ((i=0; i<$num_epoch; i++)); do
  check_file="response_famassoc/r_epoch${i}_tx${gamma}_${tau_test_1}ms_5_4x10_${si_1}_64_8_8_${postfix}.npy"
  if [ ! -f "$check_file" ]; then
    wrp_path="weights_famassoc/weight_epoch${i}_tx${gamma}_${tau_train}ms_5_4x10_64_8_8_${postfix}.npy"
    logfile=${cwd}/log_assoc/test_trained_epoch$i.txt
    echo "Running epoch$i testing ... "
    srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
      module load anaconda3
      source activate fam
      cd ${cwd} 
      python test.py --r_in_path ${r_in_path} --out_path ${out_path_test} --task ${task} --tau_stim ${tau_test_1} --save_every_stim ${si_1} --wie ${wie} --set_wrp ${swrp} --wrp_path ${wrp_path} --postfix ${postfix} --get_BCM_threshold ${BCM_thresh}
EOF
    pids_3+=($!)
  
  else
    echo "Output file exists for epoch$i testing, skipped."
  fi
done

# test2
check_file="response_famassoc/r_epoch4_tx${gamma}_${tau_test_2}ms_5_4x10_${si_2}_64_8_8_${postfix}.npy"
if [ ! -f "$check_file" ]; then
  wrp_path="weights_famassoc/weight_epoch4_tx${gamma}_${tau_train}ms_5_4x10_64_8_8_${postfix}.npy"
  logfile=${cwd}/log_assoc/test_trained_epoch4_2.txt
  echo "Running epoch4 testing2 ... "
  srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python test.py --r_in_path ${r_in_path} --out_path ${out_path_test} --task ${task} --tau_stim ${tau_test_2} --save_every_stim ${si_2} --wie ${wie} --set_wrp ${swrp} --wrp_path ${wrp_path} --postfix ${postfix} --get_BCM_threshold ${BCM_thresh}
EOF
  pids_3+=($!)

else
  echo "Output file exists for epoch4 testing2, skipped."
fi

for pid in ${pids_3[@]}; do
    wait $pid
done

# find projection distance
wrp_path="weights_famassoc/weight_epoch0_tx${gamma}_${tau_train}ms_5_4x10_64_8_8_${postfix}.npy"
response_path="response_famassoc/r_epoch0_tx${gamma}_${tau_test_1}ms_5_4x10_${si_1}_64_8_8_${postfix}.npy"
response_path_pre="response_famassoc/r_pre_${tau_test_1}ms_5_4x10_${si_1}_64_8_8_${postfix}.npy"

jac_epoch='epoch4'
jac_img=1
jac_level='10%'
jac_sample=0

logfile=${cwd}/log_assoc/find_proj_dist.txt
echo "Computing dynamical mode and projections ... "
srun -p cpu -n1 --mem=80GB --time=8-00:00:00 --pty bash << EOF &> $logfile &
  module load anaconda3
  source activate fam
  cd ${cwd}
  python find_proj_dist.py --input_path ${r_in_path} --weight_path ${wrp_path} --response_path ${response_path} --response_path_pre ${response_path_pre} --save_jac ${jac_epoch} ${jac_img} ${jac_level} ${jac_sample}
EOF
pid4=$!
wait $pid4
