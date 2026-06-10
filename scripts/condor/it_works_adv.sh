#!/bin/bash -x

set -e
cd ..
# cd ~/../../research/harris/robin/noisy-ann/scripts/condor
# Start timer to track how long it takes the script to run 
start=$(date +%s)

# Define the arguments
COVROT=0
TRIAL=$1
COVADV=$2
EPS=$3
TRACE=2.0
DATASET="Fashion"
ENSMBL=10   # Number of ensembles
NL=1    # Noisy layer

# Define the commands
COMMANDS=(
  "./mnist_train.py --noisy zero"
  "./mnist_train.py --noisy full"
  "./mnist_train.py --noisy identity"
  "./mnist_train.py --noisy diagonal"
  "./eval_robustness.py --noisy full"
  "./eval_robustness.py --noisy zero"
  "./eval_robustness.py --noisy identity"
  "./eval_robustness.py --noisy diagonal"
)

SAVEDIR="../saved/it_works/trial${TRIAL}/covadv_${COVADV}_EPS_${EPS}_${TRACE}/"
mkdir -p $SAVEDIR

DT=$(date)
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NEW RUN $DT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 
# Execute the commands to create the base model and save adversarial data
./mnist_train.py --savedir $SAVEDIR --dataset $DATASET 

./compute_covariance.py --savedir $SAVEDIR --noisy-layer $NL --covrot $COVROT --dataset $DATASET --covcorr-sev $SEVERITY --corr-sev $SEVERITY --covtrans $TRANSFORM --transform $TRANSFORM

# Loop over the commands and append the arguments
for CMD in "${COMMANDS[@]}"; do
    FULL_CMD="$CMD --savedir $SAVEDIR --reinit --covadv $COVADV --covrot $COVROT --adv-cov-eps $EPS --eps $EPS --dataset $DATASET --noisy-layer $NL --n-ensemble $ENSMBL --trace-scale $TRACE --eval-mod adv"
    eval $FULL_CMD
done

# Print time elapsed 
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds" 