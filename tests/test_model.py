import torch
# from model import MyAwesomeModel
import pytest
from src.mnist_project.model import MyAwesomeModel
from tests import _PATH_DATA
import os.path


# @pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
@pytest.mark.parametrize("batch_size", [32, 64])
def test_model(batch_size: int):
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10), "Expected output shape to be [1, 10]"


# @pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1, 2, 3))
    with pytest.raises(ValueError, match=r'Expected each sample to have shape \(1, 28, 28\)'):
        model(torch.randn(1, 1, 28, 29))
