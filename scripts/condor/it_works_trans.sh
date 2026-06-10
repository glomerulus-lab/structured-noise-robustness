#!/bin/bash -x

set -e
cd ..

# Start timer to track how long it takes the script to run 
start=$(date +%s)

# Define the arguments
TRIAL=$1
TRANSFORM=$2
TRANSFORM_SCALE=$3
TRACE=0.5
DATASET="Fashion"
ENSMBL=10   # Number of ensembles
NL=1    # Noisy layer

# Define the commands
COMMANDS=(
  "./mnist_train.py --noisy zero"
  "./mnist_train.py --noisy full"
  "./mnist_train.py --noisy identity"
  "./mnist_train.py --noisy diagonal"
  "./eval_robustness.py --noisy full --transform $TRANSFORM"
  "./eval_robustness.py --noisy zero --transform $TRANSFORM"
  "./eval_robustness.py --noisy identity --transform $TRANSFORM"
  "./eval_robustness.py --noisy diagonal --transform $TRANSFORM"
)

SAVEDIR="../saved/it_works/trial${TRIAL}/${TRANSFORM}_${TRANSFORM_SCALE}/"
DATADIR="../saved/it_works/trial${TRIAL}/mod_data/"

mkdir -p $SAVEDIR

DT=$(date)
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NEW RUN $DT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 
# Execute the commands to create the base model and save adversarial data
./mnist_train.py --savedir $SAVEDIR --dataset $DATASET 

./compute_covariance.py --savedir $SAVEDIR --moddatadir $DATADIR --noisy-layer $NL --dataset $DATASET --covtrans $TRANSFORM --transform $TRANSFORM --covtrans-scale $TRANSFORM_SCALE --transform-scale $TRANSFORM_SCALE

# Loop over the commands and append the arguments
for CMD in "${COMMANDS[@]}"; do
    FULL_CMD="$CMD --savedir $SAVEDIR --moddatadir $DATADIR --reinit --dataset $DATASET --noisy-layer $NL --n-ensemble $ENSMBL --trace-scale $TRACE --covtrans $TRANSFORM --transform-scale $TRANSFORM_SCALE --covtrans-scale $TRANSFORM_SCALE --eval-mod matched"
    eval $FULL_CMD
done

# Print time elapsed 
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds" 