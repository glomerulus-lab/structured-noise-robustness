#!/bin/bash -x

set -e
cd ..
# cd ~/../../research/harris/robin/noisy-ann/scripts/condor
# Start timer to track how long it takes the script to run 
start=$(date +%s)

# Define the arguments
COVROT=0
COVADV=None
EPS=None
EXPERIMENT=$1
TRIAL=$2
TRANSFORM_SCALE=$3
SEVERITY=$4
TRACE=0.5
DATASET="Fashion"
ENSMBL=10   # Number of ensembles
NL=1    # Noisy layer

DATADIR="../saved/${EXPERIMENT}/trial${TRIAL}/mod_data/"

mkdir -p $DATADIR

DT=$(date)
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NEW RUN $DT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" 

./gen_mod_data.py --eval-mod all --moddatadir $DATADIR --dataset $DATASET --corr-sev $SEVERITY --transform-scale $TRANSFORM_SCALE 

# Print time elapsed 
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds" 