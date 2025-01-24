#!/bin/bash
cifar_path=""  # cifar saving path
cwd=""  # project directory

# ====================================================================================================
# params
task="effects"
wies_sweep=(10 15 20 25 30 35 40 45 50)
wie_f=30
postfix="v3_cifar"
tau_train=150
gamma=30
num_epoch=5
post_only=1
tau_test=500
si=250
si_sweep=25
# ====================================================================================================

out_path_train="weights_fameff"
out_path_test="response_fameff"
out_path_train_sweep='weights_fameff_wiesweep'
out_path_test_sweep='response_fameff_wiesweep'

mkdir -p ${cwd}/log_effects

# get input
check_file="input_fameff/r_in_cifarPure_abs_500_64_8_8.npy"
if [ ! -f "$check_file" ]; then
logfile=${cwd}/log_effects/input_gen.txt
  echo "Generating feedforward inputs ... "
  srun -p cpu -n1 --mem=10GB --time=1:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python input_cifar.py --cifar_path ${cifar_path} --task ${task}
EOF
  pid_0=$!
  wait $pid_0

else
  echo "Input file existed, skipped."
fi

r_in_path="input_fameff/r_in_cifarPure_abs_500_64_8_8.npy"

# get pre-trained response and BCM threshold
pids_1=()
swrp=0
wrp_path='.'
BCM_thresh=1

for wie in "${wies_sweep[@]}"; do
  check_file="response_fameff_wiesweep/r_pre_${tau_test}ms_500_${si_sweep}_64_8_8_${wie}_${postfix}.npy"
  if [ ! -f "$check_file" ]; then
    logfile=${cwd}/log_effects/test_pre-trained_wiesweep_wie=${wie}.txt
    echo "Running pre-trained testing, wiesweep, wie=${wie} ... "
    srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
      module load anaconda3
      source activate fam
      cd ${cwd}
      python test.py --r_in_path ${r_in_path} --out_path ${out_path_test_sweep} --task ${task} --tau_stim ${tau_test} --save_every_stim ${si_sweep} --wie ${wie} --set_wrp ${swrp} --wrp_path ${wrp_path} --postfix ${postfix} --get_BCM_threshold ${BCM_thresh}
EOF
    pids_1+=($!)
  
  else
    echo "Output file exists for pre-trained testing, wiesweep, wie=${wie}, skipped."
  fi
done

BCM_thresh=0
check_file="response_fameff/r_pre_${tau_test}ms_500_${si}_64_8_8_${wie_f}_${postfix}.npy"
if [ ! -f "$check_file" ]; then
  logfile=${cwd}/log_effects/test_pre-trained_wie=${wie_f}.txt
  echo "Running pre-trained testing, wie=${wie_f} ... "
  srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python test.py --r_in_path ${r_in_path} --out_path ${out_path_test} --task ${task} --tau_stim ${tau_test} --save_every_stim ${si} --wie ${wie_f} --set_wrp ${swrp} --wrp_path ${wrp_path} --postfix ${postfix} --get_BCM_threshold ${BCM_thresh}
EOF
  pids_1+=($!)

else
  echo "Output file exists for pre-trained testing, wie=${wie_f}, skipped."
fi

for pid in ${pids_1[@]}; do
    wait $pid
done

## run training
pids_2=()
for wie in "${wies_sweep[@]}"; do
  check_file="weights_fameff_wiesweep/weight_epoch4_tx${gamma}_${tau_train}ms_500_64_8_8_${wie}_${postfix}.npy"
  if [ ! -f "$check_file" ]; then
    theta_path="input_fameff/BCM_theta_cifarPure_abs_500_64_8_8_ys_${wie}_${postfix}.npy"
    logfile=${cwd}/log_effects/train_wiesweep_wie=${wie}.txt
    echo "Running training, wiesweep, wie=${wie} ... "
    srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
      module load anaconda3
      source activate fam
      cd ${cwd}
      python train.py --r_in_path ${r_in_path} --theta_path ${theta_path} --out_path ${out_path_train_sweep} --task ${task} --gamma ${gamma} --tau_stim ${tau_train} --wie ${wie} --epoch ${num_epoch} --postfix ${postfix} --post_only ${post_only}
EOF
    pids_2+=($!)
  
  else
    echo "Output file exists for training, wiesweep, wie=${wie}, skipped."
  fi
done

check_file="weights_fameff/weight_epoch4_tx${gamma}_${tau_train}ms_500_64_8_8_${wie_f}_${postfix}.npy"
if [ ! -f "$check_file" ]; then
  theta_path="input_fameff/BCM_theta_cifarPure_abs_500_64_8_8_ys_${wie_f}_${postfix}.npy"
  logfile=${cwd}/log_effects/train_wie=${wie_f}.txt
  echo "Running training, wie=${wie_f} ... "
  srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python train.py --r_in_path ${r_in_path} --theta_path ${theta_path} --out_path ${out_path_train} --task ${task} --gamma ${gamma} --tau_stim ${tau_train} --wie ${wie_f} --epoch ${num_epoch} --postfix ${postfix} --post_only ${post_only}
EOF
  pids_2+=($!)

else
  echo "Output file exists for training, wie=${wie_f}, skipped."
fi

for pid in ${pids_2[@]}; do
    wait $pid
done

# run testing
pids_3=()
for wie in "${wies_sweep[@]}"; do
  check_file="response_fameff_wiesweep/r_epoch4_tx${gamma}_${tau_test}ms_500_${si_sweep}_64_8_8_${wie}_${postfix}.npy"
  if [ ! -f "$check_file" ]; then
    swrp=1
    BCM_thresh=0
    wrp_path="weights_fameff_wiesweep/weight_epoch4_tx${gamma}_${tau_train}ms_500_64_8_8_${wie}_${postfix}.npy"

    logfile=${cwd}/log_effects/test_trained_epoch4_wiesweep_wie=${wie}.txt
    echo "Running epoch4 testing, wiesweep, wie=${wie} ... "
    srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
      module load anaconda3
      source activate fam
      cd ${cwd}
      python test.py --r_in_path ${r_in_path} --out_path ${out_path_test_sweep} --task ${task} --tau_stim ${tau_test} --save_every_stim ${si_sweep} --wie ${wie} --set_wrp ${swrp} --wrp_path ${wrp_path} --postfix ${postfix} --get_BCM_threshold ${BCM_thresh}
EOF
    pids_3+=($!)
  
  else
    echo "Output file exists for epoch4 testing, wiesweep, wie=${wie}, skipped."
  fi
done

check_file="response_fameff/r_epoch4_tx${gamma}_${tau_test}ms_500_${si}_64_8_8_${wie_f}_${postfix}.npy"
if [ ! -f "$check_file" ]; then
  swrp=1
  BCM_thresh=0
  wrp_path="weights_fameff/weight_epoch4_tx${gamma}_${tau_train}ms_500_64_8_8_${wie_f}_${postfix}.npy"

  logfile=${cwd}/log_effects/test_trained_epoch4_wie=${wie_f}.txt
  echo "Running epoch4 testing, wie=${wie_f} ... "
  srun -p gpu -n1 --gpus=1 --mem-per-gpu=12GB --mem=40GB --time=8:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python test.py --r_in_path ${r_in_path} --out_path ${out_path_test} --task ${task} --tau_stim ${tau_test} --save_every_stim ${si} --wie ${wie_f} --set_wrp ${swrp} --wrp_path ${wrp_path} --postfix ${postfix} --get_BCM_threshold ${BCM_thresh}
EOF
  pids_3+=($!)

else
  echo "Output file exists for epoch4 testing, wie=${wie_f}, skipped."
fi

for pid in ${pids_3[@]}; do
    wait $pid
done
