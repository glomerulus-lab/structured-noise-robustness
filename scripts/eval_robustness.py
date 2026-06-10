#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import joblib
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import SquareAttack
from art.attacks.evasion import AutoProjectedGradientDescent
from art.defences.preprocessor import GaussianAugmentation
from param_utils import Params
from data_utils import load_MNIST_ART_format

# Runs a single attack using the given features x and targets y, returns accuracy
# If attack is None, evaluates model on x
def eval_model_acc(classifier, attack, x, y, n_ensemble=1):
    if attack is None:
        x_adv = x
    else: 
        x_adv = attack.generate(x=x, y=y)
    pred = classifier.predict(x_adv)
    for _ in range(n_ensemble-1):
        pred += classifier.predict(x_adv)
    pred /= n_ensemble
    acc = np.sum(np.argmax(pred, axis=1) == y) / len(y)
    return acc.item(), x_adv

# Attack model & save experiment details and accuracy to 'results' dataframe
# If attack is None, then it evaluates the model on x with no modifications. 
def attack_model(attack, attack_name, eps, args, classifier, noise_type, dataset_str, x, y, n_ensemble=1, transform=None):
        result = {"dataset": dataset_str, "attack": attack_name, "noise": noise_type, "eps": eps, "covrot": args.covrot, "covadv": args.covadv, "ac_eps":args.adv_cov_eps, "alpha":args.alpha, "beta":args.beta, "trace":args.trace_scale, "noisy_layer": args.noisy_layer, "n_ensemble": n_ensemble, "arch": args.arch, "transform": transform, "transform_scale": args.transform_scale, "covtrans": args.covtrans, "covtrans_scale": args.covtrans_scale, "corr_sev": args.corr_sev, "covcorr_sev": args.covcorr_sev}
        if args.rotate is None:
            result.update({"rotate": 0})
        else:
            result.update({"rotate": args.rotate})

        acc_adv, x_adv = eval_model_acc(classifier, attack, x, y, n_ensemble)
        result.update({"accuracy": acc_adv})

        return result, x_adv

# Evaluate accuracy of model before wrapping in pytorch classifier
# Used to confirm that wrapper did not alter model performance 
def eval_unwrapped_model_acc(model, x, y, device):
    total = len(y)
    with torch.no_grad():
        outputs = model(torch.from_numpy(x.reshape((-1, 1, 28, 28))).to(device))
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        correct = (predicted == y).sum().item()

    acc = correct / total
    return acc

# Save <num_examples> modified data points as pngs
# 'start_index' is the index of x (dataset) that we start at
def view_modified_data(x_adv, attack_name, eps, num_examples, params, start_index=0, store_in_trial_folder=True):
    args=params.args
    if store_in_trial_folder:
        folder_path = f"{params.savedir}/adv_examples/"
        filename = f"{folder_path}{attack_name}_eps_{eps}"
    else: 
        folder_path = f"{params.savedir}/../../adv_examples/"
        # This filename is absolutely ridiculous, but it prevents name collisions
        filename = (f"{folder_path}{attack_name}_eps-{eps}_trans_{args.transform}_trans_scale_{args.transform_scale}_sev_{args.corr_sev}_rot-{args.rotate}_covrot-{args.covrot}_{params.noise.replace(' ', '')}_covadv-{args.covadv}_trace-{args.trace_scale}_noisy_layer--{params.noisy_layer}_nensemble--{args.n_ensemble}_arch_{args.arch}_corrsev_{args.covcorr_sev}")

    if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    for i in range(0, num_examples):
        plt.imshow(x_adv[i+start_index][0], cmap="gray")
        plt.title(f"{attack_name}, eps={eps}, scale={args.transform_scale}, sev={args.corr_sev}")
        plt.savefig(f"{filename}_{i}.png")



def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    params = Params(args_needed=['rotate', 'noisy', 'covrot', 'covadv', 'iter', 'eps'])
    args = params.args
    eps = args.eps
    n_ensemble = args.n_ensemble

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_MNIST_ART_format(params)

    if args.noisy == True or args.noisy == "full": # noise based on full covariance matrix
        noise_type = "Full Cov"
    elif args.noisy == "diagonal":
        noise_type = "Diagonal"
    elif args.noisy == "identity":
        noise_type = "Identity"
    else: 
        noise_type = "No Noise"

    # Load model 
    model = params.Net(params, device=device)
    model.to(device)
    model_fp = params.model_filename()
    print("Loading model parameters from ", model_fp)
    model.load_state_dict(torch.load(model_fp))
    model.eval()

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # If gaussian_aug is being applied, create preprocessor
    if args.transform == "gaussian_aug":
        preprocessor = GaussianAugmentation(
            sigma=args.transform_scale, 
            augmentation=False, 
            clip_values=(0.0, 1.0), 
            apply_predict=True
            )
    else:
        preprocessor = None

    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        preprocessing_defences=preprocessor
    )

    # Evaluate model against benign data & modifications specified by 
    # command line arguments
    # Accumulate model performance data in a list of dictionaries 
    performance = []

    # Benign
    if args.eval_on_train:
        benign_perf_train, _ = attack_model(None, "benign", eps, args, classifier, noise_type, "train", x_train, y_train, n_ensemble)
        performance.append(benign_perf_train)
    else: 
         benign_perf_train = "N/A"
    benign_perf_test, _ = attack_model(None, "benign", eps, args, classifier, noise_type, "test", x_test, y_test, n_ensemble)
    performance.append(benign_perf_test)
    print(f"BENIGN\t train_acc: {benign_perf_train}\t test_acc: {benign_perf_test}")
    # view_modified_data(x_train, "benign", eps, 5, params, 5)

    if args.eval_mod in ["all", "adv", "FGM"]:
        # FastGradientMethod
        fgm = FastGradientMethod(estimator=classifier, eps=eps)
        if args.eval_on_train:
            fgm_perf_train, x_fgm_train = attack_model(fgm, "FGM", eps, args, classifier, noise_type, "train", x_train, y_train, n_ensemble)
            performance.append(fgm_perf_train)
        else: 
            fgm_perf_train = "N/A"
        fgm_perf_test, x_fgm_test = attack_model(fgm, "FGM", eps, args, classifier, noise_type, "test", x_test, y_test, n_ensemble)
        performance.append(fgm_perf_test)
        print(f"FGM\t train_acc: {fgm_perf_train}\t test_acc: {fgm_perf_test}")
        # view_modified_data(x_fgm_train, "FGM", eps, 5, params, 5)

    if args.eval_mod in ["all", "adv", "PGD"]:
        # ProjectedGradientDescent
        pgd = ProjectedGradientDescent(estimator=classifier, eps=eps, verbose=False) 
        if args.eval_on_train:
            pgd_perf_train, x_pgd_train = attack_model(pgd, "PGD", eps, args, classifier, noise_type, "train", x_train, y_train, n_ensemble)
            performance.append(pgd_perf_train)
        else: 
            pgd_perf_train = "N/A"
        pgd_perf_test, x_pgd_test = attack_model(pgd, "PGD", eps, args, classifier, noise_type, "test", x_test, y_test, n_ensemble)
        performance.append(pgd_perf_test)
        print(f"PGD\t train_acc: {pgd_perf_train}\t test_acc: {pgd_perf_test}")
        # view_modified_data(x_pgd_train, "PGD", eps, 5, params, 5)
        
    if args.eval_mod in ["all", "adv", "Square"]:
        # Square
        square = SquareAttack(estimator=classifier, eps=eps, verbose=False) 
        if args.eval_on_train:
            square_perf_train, x_square_train = attack_model(square, "Square", eps, args, classifier, noise_type, "train", x_train, y_train, n_ensemble)
            performance.append(square_perf_train)
        else: 
            square_perf_train = "N/A"
        square_perf_test, x_square_test = attack_model(square, "Square", eps, args, classifier, noise_type, "test", x_test, y_test, n_ensemble)
        performance.append(square_perf_test)
        print(f"Square\t train_acc: {square_perf_train}\t test_acc: {square_perf_test}")
        # view_modified_data(x_square_test, "Square", eps, 5, params, 5)

    if args.eval_mod in ["all", "adv", "AutoPGD"]:
        # Auto PGD
        autoPGD = AutoProjectedGradientDescent(estimator=classifier, eps=eps, verbose=False) 
        if args.eval_on_train:
            autoPGD_perf_train, x_autoPGD_train = attack_model(autoPGD, "AutoPGD", eps, args, classifier, noise_type, "train", x_train, y_train, n_ensemble)
            performance.append(autoPGD_perf_train)
        else: 
            autoPGD_perf_train = "N/A"
        autoPGD_perf_test, x_autoPGD_test = attack_model(autoPGD, "AutoPGD", eps, args, classifier, noise_type, "test", x_test, y_test, n_ensemble)
        performance.append(autoPGD_perf_test)
        print(f"AutoPGD\t train_acc: {autoPGD_perf_train}\t test_acc: {autoPGD_perf_test}")
        # view_modified_data(autoPGD_perf_test, "AutoPGD", eps, 5, params, 5)

    if args.eval_mod in ["all", "matched", "transform"]:
        # Evaluate model against non-adversarial modifications 
        if args.eval_mod == "matched": 
            # Evaluate model against the same modification that was
            # used to generate the noise covariance matrix 
            transforms = [args.transform]
        else: 
            # Evaluate model against all non-adversarial modifications 
            transforms = ["impulse_noise", "gaussian_noise", "contrast", "brightness", "motion_blur", "snow", "elastic","perspective", "obstruction", "rotate"] 
        
        for transform in transforms:
            # Load the transformed MNIST dataset
            (x_trans_train, y_trans_train), (x_trans_test, y_trans_test), min_pixel_value, max_pixel_value = load_MNIST_ART_format(params, transform=transform)
            if args.eval_on_train:
                trans_perf_train, x_trans_train = attack_model(None, "transformed", eps, args, classifier, noise_type, "train", x_trans_train, y_trans_train, n_ensemble, transform=transform)
                performance.append(trans_perf_train)
            else: 
                trans_perf_train = "N/A"
            trans_perf_test, x_trans_test = attack_model(None, "transformed", eps, args, classifier, noise_type, "test", x_trans_test, y_trans_test, n_ensemble, transform=transform)
            performance.append(trans_perf_test)
            print(f"{transform}\t train_acc: {trans_perf_train}\t test_acc: {trans_perf_test}")
            # view_modified_data(x_trans_test, "trans", eps, 5, params, 5, store_in_trial_folder=False)

    # Save performance data to a file
    performance_df = pd.DataFrame.from_records(performance)
    print(performance_df)
    if args.rotate is None:
        rotation = 0
    else:
        rotation = args.rotate
        
    if not os.path.exists(f"{params.savedir}/attack_perf/"):
            os.makedirs(f"{params.savedir}/attack_perf/")
    performance_df.to_pickle(f"{params.savedir}/attack_perf/{noise_type.replace(' ', '')}_rot-{rotation}_covrot-{args.covrot}_covadv-{args.covadv}_eps-{eps}_alpha-{args.alpha}_beta--{args.beta}_trace-{args.trace_scale}_noisy_layer--{params.noisy_layer}_nensemble--{n_ensemble}_arch_{args.arch}_trans_{args.transform}_trans_scale_{args.transform_scale}_corrsev_{args.corr_sev}_{args.covcorr_sev}.pkl")

if __name__ == '__main__':
    main()