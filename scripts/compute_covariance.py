#!/usr/bin/env python3

import numpy as np
import numpy.linalg as la
import torch
import os 
from param_utils import Params
from data_utils import MnistData

def main():
    params = Params(args_needed=['covrot', 'covadv', 'iter'])

    # Load the original dataset
    data = MnistData(params, traineval=True)

    # Load data and create the modified training dataset
    data_mod = MnistData(params, 
                         rotation_angle=params.args.covrot, 
                         attack=params.args.covadv, 
                         transform=params.args.covtrans, 
                         transform_scale=params.args.covtrans_scale, 
                         corr_sev=params.args.covcorr_sev, 
                         random=False,
                         traineval=True)
    # Load the network
    net = params.Net(params)
    net.load_state_dict(torch.load(params.base_model_filename()))
    net.eval()

    # Compute covariance matrix on forward pass
    images, _ = next(iter(data.loader))
    images_rot, _ = next(iter(data_mod.loader))

    with torch.no_grad():
        # Get response to the modified images
        net(images_rot.reshape((-1, 1, 28, 28)))  # Forward pass
        activn_mod = net.noisy_layer_output().detach().cpu().numpy().copy()
        # Create copy to ensure we aren't working with a reference to the tensor

        # Get response to the original images
        net(images.reshape((-1, 1, 28, 28)))
        activn = net.noisy_layer_output().detach().cpu().numpy()

    # We are relying on the fact that the modification functions as well as the
    # dataloader (with traineval=True) do not change the ordering of samples
    diff = activn_mod - activn
    cov = np.cov(diff.T)
    # Add 10^-4 to the diagonal of the covariance matrix to ensure it is well behaved 
    cov = cov + (np.eye(cov.shape[0]) * 0.0001)


    # Check basic statistics of the covariance matrix
    lamda = la.eigvalsh(cov)
    print('Eigenvalues of covariance matrix:')
    print(np.sort(lamda))

    # Save the covariance matrix, use tmp file to ensure atomicity
    tmp = params.cov_filename().replace(".npy", "_TEMP.npy")
    np.save(tmp, cov)
    os.replace(tmp, params.cov_filename())

if __name__ == '__main__':
    main()