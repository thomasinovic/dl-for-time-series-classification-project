"""
Loads the LSST dataset using `tslearn`
reshapes the numpy arrays into the `(n_samples, n_channels, n_timesteps)` format expected by the feature extractors.
"""

import numpy as np
from tslearn.datasets import UCR_UEA_datasets

def load_lsst_data():
    """Load the LSST dataset and format dimensions."""
    # Load the LSST dataset from UEA archive
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")
    
    # Swap axes to match (n_samples, n_channels, n_timesteps) format
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    
    # Flatten labels for scikit-learn
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    
    return X_train, y_train, X_test, y_test 