#!/bin/bash -x

set -e
cd ..
# cd ~/../../research/harris/robin/noisy-ann/scripts/condor
# Start timer to track how long it takes the script to run 
start=$(date +%s)

# Define the arguments
COVROT=0
COVADV=$4
TRIAL=$1
EPS=$2
TRACE=$3
ATTACK=$4

DATASET="Fashion"
ENSMBL=10   # Number of ensembles
NL=1    # Noisy layer

# Define the commands
COMMANDS=(
  "./mnist_train.py --noisy zero"
  "./mnist_train.py --noisy full"
  "./mnist_train.py --noisy identity"
  "./mnist_train.py --noisy diagonal"
)


SAVEDIR="../saved/training_curve/${ATTACK}_${EPS}_${TRACE}/trial${TRIAL}/"
mkdir -p $SAVEDIR

DT=$(date)
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NEW RUN $DT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 
# Execute the commands to create the base model and save adversarial data
./mnist_train.py --savedir $SAVEDIR --dataset $DATASET --training-curve

if [ "$COVADV" == "None" ]
then
./compute_covariance.py --savedir $SAVEDIR --noisy-layer $NL --covrot $COVROT --dataset $DATASET
else
./gen_adv_data.py --savedir $SAVEDIR --covrot $COVROT --eps $EPS --adv-cov-eps $EPS --dataset $DATASET --adv-data $COVADV
./compute_covariance.py --savedir $SAVEDIR --noisy-layer $NL --covadv $COVADV --covrot $COVROT --adv-cov-eps $EPS --dataset $DATASET
fi

# Loop over the commands and append the arguments
for CMD in "${COMMANDS[@]}"; do
    FULL_CMD="$CMD --training-curve --savedir $SAVEDIR --reinit --covadv $COVADV --eps $EPS --covrot $COVROT --dataset $DATASET --noisy-layer $NL --n-ensemble $ENSMBL --trace-scale $TRACE --adv-cov-eps $EPS --covadv $COVADV --eval-mod matched"
    eval $FULL_CMD
done

# Print time elapsed 
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds" 