"""Unit tests for the preprocessing module."""

import pickle

from training.register import load_pickle


def test_load_pickle(tmp_path):
    """Test the load_pickle function."""
    obj = {"a": 1}
    file = tmp_path / "obj.pkl"
    with open(file, "wb") as f:
        pickle.dump(obj, f)
    loaded = load_pickle(str(file))
    assert loaded == obj
