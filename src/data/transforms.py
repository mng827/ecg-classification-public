# Reference:
# Raghu, Aniruddh, et al. "Data augmentation for electrocardiograms."
# Conference on Health, Inference, and Learning. PMLR, 2022.
# Paper: https://arxiv.org/pdf/2204.04360.pdf
# Code adapted from: https://github.com/aniruddhraghu/ecg_aug

import numpy as np


class GaussianNoise:
    """ Adds gaussian noise to the signal
    """
    def __init__(self, magnitude_range=(0, 0.02), prob=0.5):
        self.magnitude_range = magnitude_range
        self.prob = prob

    def __call__(self, x):
        do_transform = np.random.random() < self.prob
        if do_transform:
            magnitude = np.random.random() * (self.magnitude_range[1] - self.magnitude_range[0]) + self.magnitude_range[0]
            return gaussian_noise(x, magnitude)
        else:
            return x


class RandomMask:
    """ Randomly set a certain fraction of the input to 0
    """
    def __init__(self, mask_frac_range=(0, 0.05), prob=0.5):
        self.mask_frac_range = mask_frac_range
        self.prob = prob

    def __call__(self, x):
        do_transform = np.random.random() < self.prob
        if do_transform:
            mask_frac = np.random.random() * (self.mask_frac_range[1] - self.mask_frac_range[0]) + self.mask_frac_range[0]
            return random_mask(x, mask_frac)
        else:
            return x


class BaselineWander:
    """ Adds a sine wave to the baseline
    """
    def __init__(self, magnitude_range=(0, 0.02), prob=0.5):
        self.magnitude_range = magnitude_range
        self.prob = prob

    def __call__(self, x):
        do_transform = np.random.random() < self.prob
        if do_transform:
            magnitude = np.random.random() * (self.magnitude_range[1] - self.magnitude_range[0]) + self.magnitude_range[0]
            return baseline_wander(x, magnitude)
        else:
            return x


def gaussian_noise(x, magnitude):
    """ Adds gaussian noise to the signal
    """
    noise = magnitude * np.random.normal(size=x.shape)
    return x + noise


def random_mask(x, mask_frac):
    """ Randomly set a certain fraction of the input to 0
    """
    x_aug = np.copy(x)
    L = x.shape[0]
    mask_points = int(L * mask_frac)
    start = np.random.randint(0, L - mask_points)
    end = start + mask_points
    x_aug[start:end] = 0
    return x_aug


def baseline_wander(x, magnitude):
    """ Adds a sine wave to the baseline
    """
    # typical breaths per second for an adult
    frequency = (np.random.normal() * 20 + 10) * 10 / 60.0
    phase = np.random.normal() * 2 * np.pi

    t = np.linspace(0, 1, x.shape[0])
    drift = magnitude * np.sin(t * frequency + phase)
    return x + drift
