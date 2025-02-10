#!/bin/bash
cwd="/user_data/weifanw/familiarity_plot"  # project directory

# ====================================================================================================
# set all parameter the same as the those used in the simulation
# different task specify different parameter!
task="assoc"
postfix="mix50%_all"

gamma=30
tau_test=500
si=250

# only used for task = 'effects'
wie_f=30 
wies_sweep=(10 15 20 25 30 35 40 45 50)
si_sweep=25

mkdir -p ${cwd}/log_effects

# plotting
logfile=${cwd}/log_effects/plot_effects.txt

if [ "$task" = "assoc" ]; then
    pre_path="response_famassoc/r_pre_${tau_test}ms_5_4x10_${si}_64_8_8_${postfix}.npy"
    post_path="response_famassoc/r_epoch4_tx${gamma}_${tau_test}ms_5_4x10_${si}_64_8_8_${postfix}.npy"
elif [ "$task" = "effects" ]; then
    pre_path="response_fameff/r_pre_${tau_test}ms_500_${si}_64_8_8_${wie_f}_${postfix}.npy"
    post_path="response_fameff/r_epoch4_tx${gamma}_${tau_test}ms_500_${si}_64_8_8_${wie_f}_${postfix}.npy"
else
    echo "Wronk task!"
fi 

# only used for task = 'effects'
pre_path_sweep="response_fameff_wiesweep/r_pre_${tau_test}ms_500_${si_sweep}_64_8_8_10_${postfix}.npy"
post_path_sweep="response_fameff_wiesweep/r_epoch4_tx${gamma}_${tau_test}ms_500_${si_sweep}_64_8_8_10_${postfix}.npy"

echo "plotting ..."
srun -p cpu -n1 --mem=20GB --time=1:00:00 --pty bash << EOF &> $logfile &
    module load anaconda3
    source activate fam
    cd ${cwd}
    python Fig2,S2.py --pre_path ${pre_path} --post_path ${post_path} --pre_path_sweep ${pre_path_sweep} --post_path_sweep ${post_path_sweep} --task ${task} --wies ${wies_sweep[@]}
EOF

wait
