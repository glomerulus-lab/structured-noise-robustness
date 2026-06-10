#!/usr/bin/env python3

import datetime as dt
import torch
import pandas as pd
import time
from param_utils import Params
from data_utils import MnistData

def train(net, data, params, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=params.adam_lr)

    # Training loop
    loss_vals = pd.DataFrame(columns=['loss', 'mb', 'epoch'])

    for epoch in range(params.num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, train_data in enumerate(data.loader):
            inputs, labels = train_data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = net(inputs.reshape((-1, 1, 28, 28)))
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0:  # Print every 100 mini-batches
                print('[%d, %d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                loss_vals.loc[len(loss_vals)] = [running_loss / 100, i+1, epoch+1]
                running_loss = 0.0
   
    if params.args.training_curve:
        # Save to file with same name as model parameters prepended with "training_loss_"
        split_model_filename = params.model_filename().rsplit("/", 1)
        tc_filename = "".join([split_model_filename[0], "/training_loss_", split_model_filename[1], ".pkl"])
        print(f"Saving training curve to {tc_filename}")
        loss_vals.to_pickle(tc_filename)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    params = Params(args_needed=['rotate', 'noisy', 'covrot', 'iter'])
    args = params.args

    if args.rotate is not None:
        data = MnistData(params, rotation_angle=args.rotate, attack=args.adv_data, random=True, transform=args.transform)
    else:
        data = MnistData(params, attack=args.adv_data, transform=args.transform)

    # Initialize the network and train
    net = params.Net(params, device=device)
    if params.args.noisy != False:
        net.load_state_dict(torch.load(params.base_model_filename()))
        net.freeze_layers()
        if args.reinit:
            net.post_noise_reinit()

    net.to(device)

    print(dt.datetime.now())
    start = time.time()
    train(net, data, params, device)
    end = time.time()
    print('Finished Training')
    print("Elapsed time: ", end-start)
    print(dt.datetime.now())

    # Save the trained model
    torch.save(net.state_dict(), params.model_filename())
