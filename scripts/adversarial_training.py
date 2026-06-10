import os 
import torch
import numpy as np
import pandas as pd
from art.estimators.classification import PyTorchClassifier
from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import ProjectedGradientDescent
from param_utils import Params
from data_utils import MnistData

def adv_train(params, model, x_train, y_train, min_pixel_value, max_pixel_value, eps, ratio):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.adam_lr)

    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    # Define attack
    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        verbose=False,
    )

    # Create & fit the trainer 
    trainer = AdversarialTrainer(classifier, attack, ratio=ratio)
    trainer.fit(x_train, y_train, batch_size=params.batch_size, nb_epochs=params.num_epochs)

    return classifier

# Evaluate accuracy on benign and attacked data 
# 'ratio' defines the ratio of adversarial examples in the training data
#  e.g. ratio = 0.9 -> 90% adversarial
def evaluate(classifier, x_train, y_train, x_test, y_test, eps, ratio, results):
    # Evaluate accuracy on benign data
    x_train_pred = np.argmax(classifier.predict(x_train), axis=1)
    x_test_pred = np.argmax(classifier.predict(x_test), axis=1)
    x_train_benign_acc = np.sum(x_train_pred == y_train) / len(y_train)
    x_test_benign_acc = np.sum(x_test_pred == y_test) / len(y_test)

    # Generate fresh adversarial data tailored to adversarially trained model
    pgd_attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        verbose=False,
    )
    x_train_attack = pgd_attack.generate(x_train, y=y_train)
    x_test_attack = pgd_attack.generate(x_test, y=y_test)
    
    # Evaluate accuracy on adversarial data
    x_train_attack_pred = np.argmax(classifier.predict(x_train_attack), axis=1)
    x_test_attack_pred = np.argmax(classifier.predict(x_test_attack), axis=1)
    x_train_adv_acc = np.sum(x_train_attack_pred == y_train) / len(y_train)
    x_test_adv_acc = np.sum(x_test_attack_pred == y_test) / len(y_test)

    # Append results to the 'results' dataframe
    results.loc[len(results)] = ["train", "benign", ratio, eps, x_train_benign_acc]
    results.loc[len(results)] = ["test", "benign", ratio, eps, x_test_benign_acc]
    results.loc[len(results)] = ["train", "PGD", ratio, eps, x_train_adv_acc]
    results.loc[len(results)] = ["test", "PGD", ratio, eps, x_test_adv_acc]

def main():
    params = Params(args_needed=['iter', 'eps'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the MNIST dataset
    x_train, y_train, x_test, y_test, min_pixel_value, max_pixel_value  = load_MNIST_ART_format(params)

    # Create the PyTorch model
    model = params.Net(params, device=device)
    model.to(device)

    # Perform adversarial training
    results = pd.DataFrame(columns=["dataset", "attack", "ratio", "eps", "accuracy"])
    classifier = adv_train(params, model, x_train, y_train, min_pixel_value, max_pixel_value, params.args.eps, params.args.ratio)
    evaluate(classifier, x_train, y_train, x_test, y_test, params.args.eps, params.args.ratio, results)
    
    # Save results
    save_dir = "../saved/art_adv_training_PGD_Fashion/"
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    results.to_pickle(f"{save_dir}eps_{params.args.eps}_ratio_{params.args.ratio}.pkl")
    
if __name__ == "__main__":
    main()