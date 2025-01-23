#!/bin/bash
cwd="/user_data/weifanw/familiarity_clean"  # project directory

# ====================================================================================================
# set all parameter the same as the those used in the simulation
task="effects"
postfix="v4_cifar"

wie=30
gamma=30
tau_test=500
si=250

# plotting
logfile=${cwd}/log_effects/plot_effects.txt
pre_path="response_fameff/r_pre_${tau_test}ms_500_${si}_64_8_8_20_${postfix}.npy"
post_path="response_fameff/r_epoch4_tx${gamma}_${tau_test}ms_500_${si}_64_8_8_20_${postfix}.npy"
echo "plotting ..."
srun -p cpu -n1 --mem=20GB --time=1:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python Fig2,S2.py --pre_path ${pre_path} --post_path=${post_path} --task ${task}
EOF

wait
