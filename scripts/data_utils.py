#!/usr/bin/env python3

import os
import sys
import random
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torchvision.transforms import v2, InterpolationMode
from skimage import img_as_ubyte, img_as_float32
from imagecorruptions import corrupt
from art.defences.preprocessor import GaussianAugmentation

class MnistData:
    def __init__(self, params, train=True, rotation_angle=0, attack=None, random=False, transform=None, transform_scale=None, corr_sev=None,
                 traineval=False):
        if rotation_angle is None:
            rotation_angle = 0
        self.train = train
        self.rotation_angle = rotation_angle
        self.random = random
        self.attack = attack
        self.type = params.args.dataset
        self.transform = transform
        if transform_scale is not None:
            self.transform_scale = transform_scale
        else:
            self.transform_scale = params.args.transform_scale

        if corr_sev is not None:
            self.corr_sev = corr_sev
        else:
            self.corr_sev = params.args.corr_sev
        
        # Apply any data modifications specified by arguments 
        if self.attack is None: 
            if rotation_angle == 0 and self.transform is None :
                # Load unmodified data 
                self._load_mnist_data() 
            elif rotation_angle != 0 and self.transform is None:
                # Load rotated data 
                # Check if MnistData file with required rotation already exists
                filename = (f'../data/{self.type}/rotated/{self.type}'
                            + ('_train' if train else '_test')
                            + ('_random' if random else '')
                            + ('_%d.pkl' % rotation_angle))
                if os.path.isfile(filename):
                    self.dataset = joblib.load(filename)
                else:
                    self._load_mnist_data()
                    self._rotate_data()
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    tmp = filename.remove_suffix(".pkl") + "_TEMP.pkl"  # use tmp file to ensure that writing is atomic 
                    joblib.dump(self.dataset, tmp)
                    os.replace(tmp, filename)  
                    
            elif self.transform is not None :
                # Apply the specified non-adversarial modification

                # True if modification is not from the imagecorruption library
                is_transform = self.transform in ["gaussian_aug", "elastic", "perspective", "obstruction", "rotate"]
                
                filename = (f'{params.moddatadir}/{self.transform}/{self.type}'
                            + ('_train' if train else '_test')
                            + ('_random' if random else '')
                            + ('_%.2f.pkl' % self.transform_scale if is_transform else '_%d.pkl' % self.corr_sev))
                if os.path.isfile(filename):
                    self.dataset = joblib.load(filename)
                else:
                    self._load_mnist_data()
                    self._transform_data()
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    tmp = filename.removesuffix(".pkl") + "_TEMP.pkl"  # use tmp file to ensure that writing is atomic 
                    joblib.dump(self.dataset, tmp)
                    os.replace(tmp, filename)  
        else:
            # Assumes file exists, raises error otherwise.
            # Adversarial data cannot be generated from this file because you must have access
            # to the model to create an attack. Use gen_adv_data.py to generate it. 
            filename = (f'{params.moddatadir}/{self.type}'
                        + ('_train' if train else '_test')
                        + ('_random' if random else '')
                        + ('_%d' % rotation_angle if rotation_angle != 0 else '')
                        + ('ac_eps_%.2f' % params.args.adv_cov_eps)
                        + (f'_{self.attack}.pkl'))
            self.dataset = joblib.load(filename)

        self.traineval = traineval
        if traineval or not train:
            # Setup for evaluating performance on training or test data
            self.loader = DataLoader(self.dataset, batch_size=len(self.dataset),
                                     shuffle=False)
        else:
            # Setup for training
            self.loader = DataLoader(self.dataset, batch_size=params.batch_size,
                                     shuffle=True)

    def _load_mnist_data(self):
        # This is an unfortunate naming collision! The transform here is NOT the same as non-adversarial data modifications
        norm_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if self.type == "MNIST":
            self.dataset = MNIST(root='../data', train=self.train,
                                download=True, transform=norm_transform)
        elif self.type == "Fashion":
            self.dataset = FashionMNIST(root='../data', train=self.train,
                                download=True, transform=norm_transform)
    def _rotate_data(self):
        """
        Rotate each image in the dataset.

        If random is true, each data point is rotated by a random angle drawn from
        Uniform[-rotation_angle, +rotation_angle].
        """

        rng = np.random.default_rng()
        rotated_images = []
        for i, (image, label) in enumerate(self.dataset):
            if (i + 1) % 10000 == 0:
                print('.', end='', flush=True)
            if self.random:
                angle = rng.uniform(-self.rotation_angle, self.rotation_angle)
            else:
                angle = self.rotation_angle

            rotated_image = transforms.functional.rotate(image, angle=angle, interpolation=transforms.InterpolationMode.BILINEAR)
            normd_rotated_img = (rotated_image-torch.min(rotated_image))/(torch.max(rotated_image)-torch.min(rotated_image))
            rotated_images.append((normd_rotated_img, label))

        self.dataset = rotated_images

    # Takes single image as numpy array, returns a torch tensor
    def _corrupt(self, image):
        # corrupt() requires images as a numpy array in ubyte format, must be at least 32x32
        # Convert images to this format and then back to a 28x28 torch tensor in float format 
        formatted_img = image.numpy()
        formatted_img = np.pad(formatted_img, ((0,), (2,), (2,)), constant_values=(0,0)).squeeze()
        formatted_img = img_as_ubyte(formatted_img)
        corrupted_img = corrupt(formatted_img, severity=self.corr_sev, corruption_name=self.transform)[2:30, 2:30, :]
        corrupted_img = img_as_float32(corrupted_img)
        corrupted_img = np.expand_dims(corrupted_img, 0)
        result = torch.from_numpy(corrupted_img).squeeze()
        result = result.permute(2, 0, 1)

        # These corruptions introduce color, so we need to convert them to grayscale
        if self.transform in ["impulse_noise", "shot_noise", "frost", "gaussian_noise"]:   
            transform = torchvision.transforms.v2.Grayscale()
            result = transform(result)
        else:
            result = result[0, :, :]    # Content of RGB channels is identical for other corruptions, so we use just one of them
        return result

    def _transform_data(self):
        """
        Transform each image in the dataset.
        """
        modified_images = []
        augmenter = None

        if self.transform == "elastic":
            elastic_alpha = 35. * self.transform_scale
            augmenter = v2.ElasticTransform(alpha=elastic_alpha)
        elif self.transform == "perspective":
            distortion_scale = 0.25 * self.transform_scale
            augmenter = v2.RandomPerspective(distortion_scale=distortion_scale, p=1.0)
        elif self.transform == "blur":
            blur_sigma = 0.7 * self.transform_scale
            augmenter = v2.GaussianBlur(kernel_size=(3, 3), sigma=blur_sigma)
        elif self.transform == "obstruction":
            augmenter = Obstruction(self.transform_scale, centered=False, color="black")
        elif self.transform == "rotate":
            rotation = 60.0 * self.transform_scale
            augmenter = v2.RandomRotation(rotation, interpolation=InterpolationMode.BILINEAR)
        elif self.transform == "gaussian_aug":
            augmenter = lambda x: torch.from_numpy(GaussianAugmentation(sigma=self.transform_scale, augmentation=False, clip_values=(0.0, 1.0))(x)[0])
        else: # Apply modifications from imagecorruption library
            def augmenter(x):
                return self._corrupt(x)
        
        # Apply modification to each image
        for i, (image, label) in enumerate(self.dataset):
            if (i + 1) % 10000 == 0:
                print('.', end='', flush=True)
            modified_img = augmenter(image)
            assert(torch.isnan(modified_img.any()) == False)
            modified_images.append((modified_img, label))
            modified_img = torch.clamp(modified_img, 0.0, 1.0)            

        self.dataset = modified_images

# Obstruction modification 
class Obstruction:
    def __init__(self, distortion_scale, base_fraction=0.40, centered=True, color="black"):
        self.distortion_scale = distortion_scale
        self.base_fraction = base_fraction
        self.centered = centered
        self.color = color

    def __call__(self, image):
        _, H, W = image.shape
        side = int(self.base_fraction * self.distortion_scale * min(H, W))
        side = max(1, min(side, min(H, W)))
        if self.centered:
            h_start = (H - side) // 2
            w_start = (W - side) // 2
        else:
            h_wiggle_room = H - side
            w_wiggle_room = W - side
            h_start = random.randint(0, h_wiggle_room)
            w_start = random.randint(0, w_wiggle_room)
        modified = image.clone()
        if self.color == "black":
            color_value = 0.0
        elif self.color == "blackOrWhite":
            color_value = random.choice([0.0, 1.0])
        elif self.color == "random":
            color_value = random.random()
        else:
            print(f"ERROR: Obstruction 'color' argument expects 'black', 'blackOrWhite', or 'random'. Recieved value {self.color}", file=sys.stderr)
        
        modified[:, h_start:h_start + side, w_start:w_start + side] = color_value
        return modified

# Helper function to load MNIST data in the format expected by Adversarial Robustness Toolbox (ART)
def load_MNIST_ART_format(params, transform=None):
    train_set = MnistData(params, rotation_angle=params.args.rotate, traineval=True, train=True, random=False, transform=transform)
    test_set = MnistData(params, rotation_angle=params.args.rotate, traineval=True, train=False, random=False, transform=transform)

    x_train, y_train = next(iter(train_set.loader))
    x_test, y_test = next(iter(test_set.loader))

    # Convert to Numpy (expected by ART)
    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)
    min_pixel_value = x_train.min()
    max_pixel_value = x_train.max()

    return (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value