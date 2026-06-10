#!/usr/bin/env python3

import math
import os
import models
from functools import wraps
import argparse

def none_or_str(value):
    if value == "None":
        return None
    return value

class Params:
    def __init__(self, args_needed, args_list=None):
        self.batch_size = 64
        self.num_epochs = 10
        self.adam_lr = 0.001
        self.num_train = 60000  # Number of training data points - hard-coded for now

        parser = argparse.ArgumentParser()

        parser.add_argument("--arch", default="v2",
            help="Neural network architecture code (see models.py).")
        parser.add_argument("--activn", default="tanh", choices=["relu", "tanh"],
            help="Nonlinearity used in the ANN activation function.")
        if "rotate" in args_needed:
            parser.add_argument("--rotate", type=int, default=None,
                help="Rotation angle in degrees to apply to training data.")
        parser.add_argument("--transform", type=str, default=None,
            choices= ["brightness", "contrast", "elastic", "gaussian_noise", "impulse_noise", \
                      "motion_blur", "obstruction", "perspective", "rotate", "snow"],
            help=("Non-adversarial modification used to alter the dataset."))
        parser.add_argument("--transform-scale", type=float, default=None,
            help=("Set the intensity of non-adversarial modifications NOT implemented using" \
                "the imagecorruption library including elastic, obstruction, perspective, " \
                "and rotate. Set to 1.0 for default intensity. Modification strength is set " \
                "to some default value (defined in data_utils.py) multiplied by " \
                "args.transform_scale."))
        parser.add_argument("--noisy", nargs="?", const=True, default=False,
            help=("Instantiate the noiseless model if false. If true, instantiate the noisy " \
                "model and load the covariance matrix computed using the rotation given by " \
                "'covrot'. If zero, use the noisy model but add no noise (i.e., to train " \
                "only post-noise layers). If diagonal, use only the diagonal of the " \
                "covariance matrix. If identity, use an identity covariance matrix."))
        parser.add_argument("--reinit", action="store_true")
        parser.add_argument("--noisy-layer", type=int, default=1,
            help="Layer of network at which to add noise.")
        parser.add_argument("--covrot", type=int, default=0,
            help=("Rotation angle used to compute the covariance matrix for adding noise " \
                "while training."))
        parser.add_argument("--pre-gen", default=False, action="store_true",
            help=("Pre-generate noise tensor at model init time."
                "If cuda is not available, noise will be dynamically generated regardless."))
        parser.add_argument("--alpha", type=float, default= None,
            help=("Used when doing a combination of full and identity covariance. " \
                "Cov = alpha*F + (1-alpha)*I where F is full and I is identity."))
        parser.add_argument("--beta", type=float, default= None,
            help=("Used when doing a combination of full and identity covariance. " \
                "If beta == None, then Cov = alpha*F + (1-alpha)*I where F is full and I is " \
                "identity. Otherwise, Cov = alpha*F + beta*I."))
        parser.add_argument("--trace-scale", type=float, default = None,
            help=("Normalize trace of covariance such that trace = trace_scale * N, where " \
                "the noise covariance matrix is NxN. Setting it to -1 is equivalent to " \
                "'None'."))
        parser.add_argument("--eval-on-train", default = False, action="store_true",
            help=("Used when running eval_robustness.py Evaluate model on modified training " \
            "set in addition to the test set."))
        parser.add_argument("--savedir", type=str, default="../saved",
            help=("Folder path to store model parameters, covariance, and performance in." \
                "Default is '../saved'."))
        parser.add_argument("--moddatadir", type=str, default=None,
            help=("Folder path to store modifed datasets in. " 
                "Set to a folder titled 'mod_data' inside of --savedir by default."))
        parser.add_argument("--covadv", type=str, default=None, 
            help=("Adversarial data used to calculate covariance matrix."))
        parser.add_argument("--covtrans", type=str, default=None,
            help=("Transform used to calculate covariance matrix."))
        parser.add_argument("--covtrans-scale", type=float, default=0.0,
            help=("Scale for transform used to calculate covariance matrix."))
        parser.add_argument("--adv-data", type=none_or_str, default=None, 
            help=("Attack to apply to the training data."))
        parser.add_argument("--eps", type=float, default=0.1,
            help=("Epsilon to use for attacks."))
        parser.add_argument("--corr-sev", type=int, default=None, 
            choices=[0, 1, 2, 3, 4, 5],
            help="Sets the strength of non-adversarial modifications used to evaluate model " \
            "robustness implemented using the imagecorruption library including brightness, " \
            "contrast, gaussian_noise, impulse_noise, motion_blur, and snow.")
        parser.add_argument("--covcorr-sev", type=int, default=None, 
            choices=[0, 1, 2, 3, 4, 5],
            help="Sets the strength of non-adversarial modifications used to generate the " \
            "noise covariance implemented using the imagecorruption library, including " \
            "brightness, contrast, gaussian_noise, impulse_noise, motion_blur, and snow.")
        parser.add_argument("--adv-cov-eps", type=float, default=0.1,
            help=("Epsilon to use for attacks when generating noise covariance. "))
        parser.add_argument("--dataset", type=str, default="MNIST", 
            choices=["MNIST", "Fashion"],
            help="Dataset to use when training and testing the model.")
        parser.add_argument("--ratio", type=float, default=None,
            help="Only used for adversarial training script. Ratio of adversarial examples " \
            "in training data.")
        parser.add_argument("--n-ensemble", type=int, default=10,
            help="Run each datapoint through the model n times and take the mode as the " \
            "model's prediction.")
        parser.add_argument("--eval-mod", type=str, default="all",
            choices=["all", "matched", "transform", "adv", "AutoPGD", "PGD", "FGM", "Square"],
            help="Specify which modifcations to evaluate the model's robustness to in " \
            "eval_robustness.py. Options include evaluating against: all modifications, " \
            "the same non-adversarial modification used to generate noise (matched), all " \
            "non-adversarial modifications (transform), all adversarial attacks (adv), or " \
            "a specific attack (AutoPGD, PGD, FGM, or Square).")
        parser.add_argument("--training-curve", default=False, action="store_true",
            help="Save training loss data to a file with the same name as model param file " \
            "prepended with 'training_loss_'.")

        if "confusion" in args_needed:
            parser.add_argument("--confusion", action="store_true",
                help=("Compute a confusion matrix. To be used while testing."))

        # If args_list is None (the default), this reads from sys.argv
        self.args = parser.parse_args(args=args_list)

        # Set default moddatadir
        if self.args.moddatadir is None:
            self.args.moddatadir = f"{self.args.savedir}/mod_data"
        
        # Ensure directory paths don't end with / 
        if self.args.savedir.endswith("/"):
            self.args.savedir = self.args.savedir.rstrip("/")
        if self.args.moddatadir.endswith("/"):
            self.args.moddatadir = self.args.moddatadir.rstrip("/")

        # Check that directories exist, if not, then create them
        if not os.path.exists(self.args.savedir):
                os.makedirs(self.args.savedir)
        if not os.path.exists(self.args.moddatadir):
                os.makedirs(self.args.moddatadir)
   
        # If rotate is 0, change to None. 
        if "rotate" in args_needed:
            if self.args.rotate == 0:
                self.args.rotate = None

        # Set trace_scale to None if -1 is passed as argument
        if self.args.trace_scale is not None and self.args.trace_scale < 0:
            self.args.trace_scale = None

        if self.args.arch == "v1":
            self.Net = models.Mnist_v1_1C5F
        elif self.args.arch == "v2":
            self.Net = models.Mnist_v2_3C3F
        elif self.args.arch == "v3":
            self.Net = models.Mnist_v3_2C3F

        self.activn = self.args.activn
        self.noisy_layer = self.args.noisy_layer
        self.savedir = self.args.savedir
        self.moddatadir = self.args.moddatadir
        self.net_name = self.Net.__name__

    def _pathify(func):
        @wraps(func)
        def convert_to_path(self):
            return os.path.join(self.savedir, func(self))
        return convert_to_path

    @property
    def net_name_nl(self):
        """Net name combined with noisy layer."""
        return self.net_name + "_N%d" % self.noisy_layer

    @_pathify
    def model_filename(self):
        args = self.args
        if (not args.noisy or args.noisy == "zero") and args.rotate is None:
            # We are in base model mode
            filename = self.net_name
        else:
            filename = self.net_name_nl
        if args.dataset != "MNIST":
            filename +=f"--{args.dataset}"
        filename += "--%s" % self.activn

        if args.noisy:
            filename += "--noisy"
            if args.noisy is not True:
                filename += "-" + args.noisy
            if args.noisy != "zero":
                filename += "--covrot-%d" % args.covrot
            if args.covadv is not None:
                filename += f"--covadv-{args.covadv}"
            if args.adv_data is not None:
                filename += f"--adv-data-{args.adv_data}"
            if not math.isclose(args.adv_cov_eps, 0.1):
                filename += "--coveps-%.2f" % args.adv_cov_eps
            if args.alpha is not None:
                filename  += "--alpha-%.2f" % args.alpha
            if args.beta is not None:
                filename  += "--beta-%.2f" % args.beta
            if args.trace_scale is not None:
                filename  += "--trace-%.2f" % args.trace_scale
            if args.covtrans is not None:
                filename += "--covtrans-%s" % args.covtrans
                filename += "--%2f" % args.covtrans_scale
            if args.covcorr_sev is not None:
                filename += "covcorr_sev-%d" % args.covcorr_sev

        if hasattr(args, "reinit") and args.reinit:
            filename += "--reinit"
        if not args.noisy or args.noisy == "zero":
            filename += ("--base" if args.rotate is None
                         else "--rot-%d" % args.rotate)
        return filename + ".pth"

    @_pathify
    def base_model_filename(self):
        # base model does not need noisy layer appended to net name
        if self.args.dataset != "MNIST":
            filename = "%s--%s--%s--base" % (self.net_name, self.args.dataset, self.activn)
        else:
            filename = "%s--%s--base" % (self.net_name, self.activn)

        return filename + ".pth"

    @_pathify
    def cov_filename(self):
        filename = "cov--%s--%s--covrot-%d--%s--cov-trans-%s-trans-scale-%.2f" % (self.net_name_nl, self.activn,
                                               self.args.covrot, self.args.covadv, self.args.covtrans, self.args.covtrans_scale)
        if self.args.dataset != "MNIST":
            filename += "--%s" % self.args.dataset
        if self.args.covcorr_sev is not None:
            filename += "covcorr_sev-%d" % self.args.covcorr_sev
        return filename + ".npy"

    @_pathify
    def perf_filename(self):
        model_filename = os.path.basename(self.model_filename())
        return "perf--" + model_filename.replace(".pth", ".pkl")

    @_pathify
    def alignment_filename(self):
        filename = ("covariance-alignment--%s--%s--%d"
                    % (self.net_name_nl, self.activn, self.args.covrot))
        return filename + ".csv"
