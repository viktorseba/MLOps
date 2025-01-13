from src.mnist_project.data import corrupt_mnist
import torch
from tests import _PATH_DATA
import os.path
import pytest


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    train_set, test_set = corrupt_mnist()
    assert len(train_set) == 30_000, "Expected 30,000 training samples"
    assert len(test_set) == 5_000, "Expected 5,000 test samples"
    for dataset in [train_set, test_set]:
        for im, target in dataset:
            assert im.shape == (1, 28, 28), "Expected shape of training data to be (784,)"  # x.shape == (1, 28, 28) or
            assert target in range(10), "Expected target to be in [0, 9]"

    train_targets = torch.unique(train_set.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all(), "Expected all targets to be present in training set"
    test_targets = torch.unique(test_set.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all(), "Expected all targets to be present in test set"
