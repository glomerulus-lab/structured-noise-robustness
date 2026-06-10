#!/bin/bash -x
# Script used to obtain data for "Non-adversarial noise covariances have limited transferability" figure
set -e
cd ..
# cd ~/../../research/harris/robin/noisy-ann/scripts/condor
# Start timer to track how long it takes the script to run 
start=$(date +%s)

# Define the arguments
TRIAL=$1
TRANSFORM=$2
COVROT=0
COVADV=None
EPS=None
TRACE=0.5
SEVERITY=5
TRANSFORM_SCALE=2.0
DATASET="Fashion"
ENSMBL=10   # Number of ensembles
NL=1    # Noisy layer

# Define the commands
COMMANDS=(
  "./mnist_train.py --noisy zero"
  "./mnist_train.py --noisy full"
  "./mnist_train.py --noisy identity"
  "./mnist_train.py --noisy diagonal"
  "./eval_robustness.py --noisy full --corr-sev $SEVERITY --transform $TRANSFORM" 
  "./eval_robustness.py --noisy zero --corr-sev $SEVERITY --transform $TRANSFORM"
  "./eval_robustness.py --noisy identity --corr-sev $SEVERITY --transform $TRANSFORM"
  "./eval_robustness.py --noisy diagonal --corr-sev $SEVERITY --transform $TRANSFORM"
)

# WARNING!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# MUST initialize all data folders before running this script by running mismatch_mod_gen_mod_data.job
SAVEDIR="../saved/mismatch_mod/trial${TRIAL}/${TRANSFORM}_cov_models/"
DATADIR="../saved/mismatch_mod/trial${TRIAL}/mod_data/"

mkdir -p $SAVEDIR

DT=$(date)
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NEW RUN $DT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 
# Execute the commands to create the base model and save adversarial data
./mnist_train.py --savedir $SAVEDIR --dataset $DATASET 

./compute_covariance.py --savedir $SAVEDIR --moddatadir $DATADIR --noisy-layer $NL --covrot $COVROT --dataset $DATASET --covtrans $TRANSFORM --covcorr-sev $SEVERITY --covtrans-scale $TRANSFORM_SCALE
# The below line is the one I used when I actually ran it, the above is a new version that I THINK wont break when I use updated compute covariance
# ./compute_covariance.py --savedir $SAVEDIR --noisy-layer $NL --covrot $COVROT --dataset $DATASET --covcorr-sev $SEVERITY --corr-sev $SEVERITY --covtrans $TRANSFORM --transform $TRANSFORM --transform-scale $TRANSFORM_SCALE --covtrans-scale $TRANSFORM_SCALE

# Loop over the commands and append the arguments
for CMD in "${COMMANDS[@]}"; do
    FULL_CMD="$CMD --savedir $SAVEDIR --reinit --covadv $COVADV --covrot $COVROT --dataset $DATASET --noisy-layer $NL --n-ensemble $ENSMBL --trace-scale $TRACE --covcorr-sev $SEVERITY --covtrans $TRANSFORM --eval-mod transform --moddatadir $DATADIR --transform-scale $TRANSFORM_SCALE --covtrans-scale $TRANSFORM_SCALE"
    eval $FULL_CMD
done

# Print time elapsed 
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds" 