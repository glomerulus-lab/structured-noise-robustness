# Usage
See bash scripts in `scrips/condor`. These scripts are set up such that they can be run on a cluster using condor, but they can also be run locally. The .txt files are used to configure the set of arguments used when running the script on the cluster, with each line containing the arguments used for a single run. 

## Example:
Navigate to /scripts/condor/ and run`./it_works_adv.sh 0 PGD 0.1` to run a noisy model with default trace & layer settings. The noise covariance will be derived from the PGD attack with eps = 0.1. 

# Description of each file in `scripts/`
## adversarial_training.py
Perform adversarial training using base model. Used to compare efficacy of structured noise injection to standard adversarial training. 
## eval_robustness.py
Run adversarial attacks against the given model and save the resulting accuracies. 
## compute_covariance.py 
Compute the noise covariance and save as a .npy file. If the data modification used is an adversarial attack, you must run `gen_adv_data.py` first. 
## data_utils.py 
Define the MnistData class and non-adversarial data modifications. 
## data_analysis.py
Code for plotting data and exporting to spreadsheets to create tables used in the paper. 
## gen_adv_data.py 
Generate adversarial data & save to a .pkl file. Must run this before `compute_covariance.py` when using an adversarial attack to obtain the noise covariance. 
## gen_mod_data.py
Generate non-adversarially modified data & save to a .pkl file. Must use this before running experiments in parallel with the same (non-adversarial) modification & strength. Otherwise, you may encounter errors due to different runs attempting to save modified data to the same filename. When using a script like `scripts/condor/gen_mod_data.sh`, the experiment folder structure is set up such that there is a one set of modified data for each trial. You do **NOT** have to run this before `compute_covariance.py` if you aren't doing multiple runs in parallel with the same modification & strength since the MnistData class will automatically generate the data if an existing .pkl file isn't found.
## mnist_train.py
Training script used for training models from models.py on MNIST or FashionMNIST. 
## models.py
Defines classes for simple CNN models intended to be trained on MnistData. 
## param_utils.py 
Defines Params class, which is used to store parameters passed in as command line arguments. Also defines default settings & filename structures. 
## save_adv_images.py 
Generate adversarial examples and save as an image. Used to obtain images for presentation & paper. 
