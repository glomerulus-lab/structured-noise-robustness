#!/usr/bin/env python3

# Modified version of code from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/examples/get_started_pytorch.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import joblib
import sys
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import DeepFool
from art.attacks.evasion import AutoProjectedGradientDescent
from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import SquareAttack
from art.defences.preprocessor import GaussianAugmentation
from param_utils import Params
from data_utils import load_MNIST_ART_format

def save_adv_data(x_adv, y, params, attack, train, random=False):
    filename = (f'{params.savedir}/mod_data/{params.args.dataset}'
                + ('_train' if train else '_test')
                + ('_random' if random else '')
                + ('rot_%d' % params.args.rotate if params.args.rotate != 0 and params.args.rotate is not None else '')
                + ('ac_eps_%.2f' % params.args.eps)
                + (f'_{attack}.pkl'))

    adv_data = [(x_adv[i], y[i]) for i in range(len(y))]

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(adv_data, filename)

# Runs a single attack using the given features  x and targets y, returns accuracy
# If attack is None, evaluates model on x
def eval_model_acc(classifier, attack, x, y):
    if attack is None:
        x_adv = x
    else: 
        x_adv = attack.generate(x=x, y=y)
    pred = classifier.predict(x_adv)
    acc = np.sum(np.argmax(pred, axis=1) == y) / len(y)
    return acc.item(), x_adv

# eps is attack size
# results_df is a dataframe that is used to accumulate results from running each attack
# if train_set or test_set is None then they are not used
def attack_model(attack, attack_name, eps, args, classifier, noise_type, dataset_str, x, y):
        result = {"dataset": dataset_str, "attack": attack_name, "noise": noise_type, "eps": eps, "covrot": args.covrot, "covadv": args.covadv}
        if args.rotate is None:
            result.update({"rotate": 0})
        else:
            result.update({"rotate": args.rotate})

        acc_adv, x_adv = eval_model_acc(classifier, attack, x, y)
        result.update({"accuracy": acc_adv})

        return result, x_adv

# Evaluate accuracy of model before wrapping in pytorch classifier
def eval_unwrapped_model_acc(model, x, y):
    total = len(y)
    with torch.no_grad():
        outputs = model(torch.from_numpy(x.reshape((-1, 1, 28, 28))).to(device))
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        correct = (predicted == y).sum().item()

    acc = correct / total
    return acc

# start_index is the index of x (dataset) that we start at
def view_adv_example(x_adv, attack_name, eps, num_examples, start_index=0):
    folder_path = "../saved/adv_examples/"

    for i in range(0, num_examples):
        plt.imshow(x_adv[i+start_index][0], cmap="gray")
        plt.title(f"{attack_name}, eps={eps}, min={x_adv[i+start_index][0].min()}, max={x_adv[i+start_index][0].max()}")
        plt.savefig(f"{folder_path}{attack_name}_eps_{eps}_{i}.png")

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    params = Params(args_needed=['rotate', 'noisy', 'covrot', 'covadv', 'adv_data', 'iter'])
    args = params.args
    eps = args.eps

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_MNIST_ART_format(params)

    if args.noisy == True: # noise based on full covariance matrix
        # model_fp = "../saved/Mnist_v3_2C3F_N1--tanh--noisy--covrot-60--reinit--0.pth"
        noise_type = "Full Cov"
    elif args.noisy == "diagonal":
        # model_fp = "../saved/Mnist_v3_2C3F_N1--tanh--noisy-diagonal--covrot-60--0.pth"
        noise_type = "Diagonal"
    elif args.noisy == "identity":
        # model_fp = "../saved/Mnist_v3_2C3F_N1--tanh--noisy-identity--covrot-60--0.pth"
        noise_type = "Identity"
    else: # no noise, no rotations
        # model_fp = "../saved/Mnist_v3_2C3F--tanh--base--0.pth"
        noise_type = "No Noise"

    model = params.Net(params)
    model.to(device)
    if noise_type == "No Noise" and args.rotate is not None:
        model_fp =f"{params.savedir}/Mnist_v3_2C3F--tanh--base--0.pth"
    else:
        model_fp = params.model_filename()
    # model_fp = "../saved/Mnist_v2_3C3F--Fashion--tanh--base--0.pth"
    print("Noise type: ", noise_type)
    print("Loading model parameters from ", model_fp)
    model.load_state_dict(torch.load(model_fp))
    model.eval()

    print("min", min_pixel_value)
    print("max", max_pixel_value)

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    augmenter = GaussianAugmentation(sigma=args.transform_scale, augmentation=False, clip_values=(0.0, 1.0), apply_predict=True)
    
    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    if args.adv_data == "FGM":
        fgm = FastGradientMethod(estimator=classifier, eps=eps)
        _, x_fgm_train = attack_model(fgm, "FGM", eps, args, classifier, noise_type, "train", x_train, y_train)
        # _, x_fgm_test = attack_model(fgm, "FGM", eps, args, classifier, noise_type, "test", x_test, y_test)
        if args.transform == "gaussian_aug":
            x_fgm_train, _ = augmenter(x_fgm_train)
        save_adv_data(x_fgm_train, y_train, params,"FGM", True)
        # save_adv_data(x_fgm_test, y_test, params,"FGM", False)
    elif args.adv_data == "PGD":
        pgd = ProjectedGradientDescent(estimator=classifier, eps=eps, verbose=False) 
        _, x_pgd_train = attack_model(pgd, "PGD", eps, args, classifier, noise_type, "train", x_train, y_train)
        # _, x_pgd_test = attack_model(pgd, "PGD", eps, args, classifier, noise_type, "test", x_test, y_test)
        if args.transform == "gaussian_aug":
            x_pgd_train, _ = augmenter(x_pgd_train)
        save_adv_data(x_pgd_train, y_train, params, "PGD", True)
        # save_adv_data(x_pgd_test, y_test, params, "PGD", False)
    elif args.adv_data == "Square":
        square = SquareAttack(estimator=classifier, eps=eps, verbose=False) 
        _, x_square_train = attack_model(square, "Square", eps, args, classifier, noise_type, "train", x_train, y_train)
        # _, x_square_test = attack_model(square, "Square", eps, args, classifier, noise_type, "test", x_test, y_test)
        if args.transform == "gaussian_aug":
            x_square_train, _ = augmenter(x_square_train)
        save_adv_data(x_square_train, y_train, params, "Square", True)
        # save_adv_data(x_square_test, y_test, params, "Square", False)
    elif args.adv_data == "AutoPGD":
        autoPGD = AutoProjectedGradientDescent(estimator=classifier, eps=eps, verbose=False) 
        _, x_autoPGD_train = attack_model(autoPGD, "AutoPGD", eps, args, classifier, noise_type, "train", x_train, y_train)
        # _, x_autoPGD_test = attack_model(autoPGD, "AutoPGD", eps, args, classifier, noise_type, "test", x_test, y_test)
        if args.transform == "gaussian_aug":
            x_autoPGD_train, _ = augmenter(x_autoPGD_train)
        save_adv_data(x_autoPGD_train, y_train, params, "AutoPGD", True)
        # save_adv_data(x_autoPGD_train, y_test, params, "AutoPGD", False)\

    else: 
        print(f"ERROR: '{args.adv_data}' is not a valid attack for gen_arg_data.py.", file=sys.stderr)
