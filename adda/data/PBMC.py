import gzip
import operator
import os
import struct
from functools import reduce

import pandas as pd
import numpy as np
import anndata
import scanpy as sc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset

@register_dataset('PBMC')
class PBMC(DatasetGroup):
    """Our human PBMC dataset

    10X PBMC, AML PBMC after feature selection with 941 features
    """

    data_files = {
            'train_data': '10X_GE.csv',
            'train_labels': '10X_GE_metadata.csv',
            'test_data': 'AML_PBMC_D11T1_GS.csv',
            'test_labels': 'AML_PBMC_D11T1_GS_metadata.csv',
            }


    def __init__(self, path=None, shuffle=True):
        DatasetGroup.__init__(self, '10XtoAML', path)
        self.data_shape = ()
        self.label_shape = ()
        self.n_clusters = None
        self.shuffle = shuffle
        self._load_datasets()

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        label_enc, train_data, train_labels = self._read_data_and_labels(
                abspaths['train_data'], abspaths['train_labels'], label_enc=None)
        label_enc, test_data, test_labels = self._read_data_and_labels(
                abspaths['test_data'], abspaths['test_labels'], label_enc=label_enc)

        assert train_data.shape[1] == test_data.shape[1] ## same feature length
        assert len(set(train_labels)) == len(set(test_labels))  ##TODO: same cluster labels for now..
        self.n_clusters = len(set(train_labels))
        self.data_shape = (train_data.shape[1])

        self.train = ImageDataset(train_data, train_labels,
                                  image_shape=self.data_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle,
                                  n_clusters=self.n_clusters)
        self.test = ImageDataset(test_data, test_labels,
                                 image_shape=self.data_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle,
                                 n_clusters=self.n_clusters)

    def _read_data_and_labels(self, data_path, label_path, label_enc=None, label_col="cell.type"):
        """Read PBMC data and labels."""
        counts = pd.read_csv(data_path, index_col=0)
        labels = pd.read_csv(label_path, index_col=0)
        adata = anndata.AnnData(X=counts, obs=labels, var=counts.columns.to_frame())
        sc.pp.normalize_per_cell(adata, min_counts=0)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=True, max_value=6)  ## center-scaled the data

        counts, labels = adata.X, adata.obs[label_col].values.ravel()
        if label_enc is None:
            label_enc = LabelEncoder()
            label_enc = label_enc.fit(labels)
        int_labels = label_enc.transform(labels)

        assert counts.shape[0] == int_labels.shape[0]  ## same length

        return label_enc, counts, int_labels
