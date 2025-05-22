# pylint: disable=<C0103>
"""Unit tests for the training module."""
import os
import pickle

import numpy as np

from train.train import run_train


def test_run_train(tmp_path):
    """Test the run_train function."""
    # Create dummy train/test data
    X_train, y_train = np.ones((2, 3)), np.array([0, 1])
    X_test, y_test = np.ones((2, 3)), np.array([0, 1])
    os.makedirs(tmp_path, exist_ok=True)
    with open(os.path.join(tmp_path, "train.pkl"), "wb") as f:
        pickle.dump((X_train, y_train), f)
    with open(os.path.join(tmp_path, "test.pkl"), "wb") as f:
        pickle.dump((X_test, y_test), f)
    # Should not raise
    run_train(data_path=str(tmp_path))
