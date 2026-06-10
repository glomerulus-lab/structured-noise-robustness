#!/usr/bin/env python3

import torch
from param_utils import Params
from data_utils import load_MNIST_ART_format
import sys

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params = Params(args_needed=['rotate', 'noisy', 'covrot', 'covadv', 'iter', 'eps'])
    args = params.args

    if args.eval_mod in ["all", "matched", "transform"]:
        if args.eval_mod == "matched":
            if args.transform is None:
                print("error: transform argument cannot be 'None' when eval_mod == matched in gen_mod_data.py", file=sys.stderr)
                exit(1)
            transforms = [args.transform]
        else: 
            if args.corr_sev is None or args.corr_sev == 0:
                transforms = ["elastic", "perspective", "obstruction", "rotate"]
            else:
                transforms = ["impulse_noise", "gaussian_noise", "contrast", "brightness", "motion_blur", "snow", "elastic", "perspective", "obstruction", "rotate"]
        
        for transform in transforms:
            # Load the transformed MNIST dataset
            _, _, _, _ = load_MNIST_ART_format(params, transform=transform)
