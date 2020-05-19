import csv
import functools
import json
import os
import random
import warnings
import pickle

import numpy as np
import torch
from pymatgen.analysis import local_env
from pymatgen.core.structure import Structure
from pymatgen.analysis.graphs import StructureGraph
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.
    !!! The dataset needs to be shuffled before using the function !!!
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool
    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.
    Parameters
    ----------
    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)
      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int
    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea = [], []
    batch_self_fea_idx, batch_nbr_fea_idx = [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx), target, cif_id)\
            in enumerate(dataset_list):

        n_i = atom_fea.shape[0]  # number of atoms for this crystal

        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_self_fea_idx.append(self_fea_idx+base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)

        crystal_atom_idx.extend([i]*n_i)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_self_fea_idx, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            torch.LongTensor(crystal_atom_idx)),\
        torch.stack(batch_target, dim=0),\
        batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array
        Parameters
        ----------
        distance: np.array shape n-d array
          A distance matrix of any shape
        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.
    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.
    Parameters
    ----------
    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:
    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...
    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.
    atom_init.json: a JSON file that stores the initialization vector for each
    element.
    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.
    Parameters
    ----------
    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The (maximum) cutoff radius for searching neighbors
    nn_method: string
        A pymatgen.analysis.local_env.NearNeighbors object used to construct
        a pymatgen.analysis.graphs.StructureGraph
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    disable_save_torch: bool
        Don't save torch files containing CIFData crystal graphs
    random_seed: int
        Random seed for shuffling the dataset
    Returns
    -------
    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """

    def __init__(self, root_dir, max_num_nbr=12, radius=8, nn_method=None,
                 dmin=0, step=0.2, disable_save_torch=False, random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius, self.nn_method = max_num_nbr, radius, nn_method
        self.disable_save_torch = disable_save_torch
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [[x.strip().replace('\ufeff', '')
                                  for x in row] for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.torch_data_path = os.path.join(self.root_dir, 'cifdata')
        if self.nn_method:
            if self.nn_method.lower() == 'minimumvirenn':
                self.nn_object = local_env.MinimumVIRENN()
            elif self.nn_method.lower() == 'voronoinn':
                self.nn_object = local_env.VoronoiNN()
            elif self.nn_method.lower() == 'jmolnn':
                self.nn_object = local_env.JmolNN()
            elif self.nn_method.lower() == 'minimumdistancenn':
                self.nn_object = local_env.MinimumDistanceNN()
            elif self.nn_method.lower() == 'minimumokeeffenn':
                self.nn_object = local_env.MinimumOKeeffeNN()
            elif self.nn_method.lower() == 'brunnernn_real':
                self.nn_object = local_env.BrunnerNN_real()
            elif self.nn_method.lower() == 'brunnernn_reciprocal':
                self.nn_object = local_env.BrunnerNN_reciprocal()
            elif self.nn_method.lower() == 'brunnernn_relative':
                self.nn_object = local_env.BrunnerNN_relative()
            elif self.nn_method.lower() == 'econnn':
                self.nn_object = local_env.EconNN()
            elif self.nn_method.lower() == 'cutoffdictnn':
                # requires a cutoff dictionary located in cgcnn/cut_off_dict.txt
                self.nn_object = local_env.CutOffDictNN(
                    cut_off_dict='cut_off_dict.txt')
            elif self.nn_method.lower() == 'critic2nn':
                self.nn_object = local_env.Critic2NN()
            elif self.nn_method.lower() == 'openbabelnn':
                self.nn_object = local_env.OpenBabelNN()
            elif self.nn_method.lower() == 'covalentbondnn':
                self.nn_object = local_env.CovalentBondNN()
            elif self.nn_method.lower() == 'crystalnn':
                self.nn_object = local_env.CrystalNN()
            else:
                raise ValueError('Invalid NN algorithm specified')
        else:
            self.nn_object = None

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        cif_id = cif_id.replace('ï»¿', '')

        target = torch.Tensor([float(target)])

        if os.path.exists(os.path.join(self.torch_data_path, cif_id+'.pkl')):
            with open(os.path.join(self.torch_data_path, cif_id+'.pkl'), 'rb') as f:
                pkl_data = pickle.load(f)
            atom_fea = pkl_data[0]
            nbr_fea = pkl_data[1]
            self_fea_idx = pkl_data[2]
            nbr_fea_idx = pkl_data[3]

        else:
            crystal = Structure.from_file(
                os.path.join(self.root_dir, cif_id+'.cif'))
            # atom features
            atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                                  for i in range(len(crystal))])

            self_fea_idx, nbr_fea_idx, nbr_fea = [], [], []
            if self.nn_object:
                graph = StructureGraph.with_local_env_strategy(
                    crystal, self.nn_object)
                all_nbrs = []
                dist_idx = -1
                for i in range(len(crystal)):
                    nbr = graph.get_connected_sites(i)
                    nbr = [nbrs for nbrs in nbr if nbrs[dist_idx] <= self.radius]
                    all_nbrs.append(nbr)
            else:
                dist_idx = 1
                all_nbrs = crystal.get_all_neighbors(
                    self.radius, include_index=True)
                all_nbrs = [sorted(nbrs, key=lambda x: x[dist_idx])
                            for nbrs in all_nbrs]
            for i, nbr in enumerate(all_nbrs):
                if len(nbr) < self.max_num_nbr:
                    warnings.warn('{} does not have enough neighbors to build graph. '
                                  'If it happens frequently, consider increasing '
                                  'radius or decreasing max_num_nbr.'.format(cif_id))

                    nbr_fea_idx.extend(list(map(lambda x: x[2], nbr)))
                    nbr_fea.extend(list(map(lambda x: x[dist_idx], nbr)))

                else:
                    nbr_fea_idx.extend(list(map(lambda x: x[2],
                                                nbr[:self.max_num_nbr])))
                    nbr_fea.extend(list(map(lambda x: x[dist_idx],
                                            nbr[:self.max_num_nbr])))

                self_fea_idx.extend([i]*min(len(nbr), self.max_num_nbr))

            nbr_fea = np.array(nbr_fea)
            nbr_fea = self.gdf.expand(nbr_fea)

            atom_fea = torch.Tensor(atom_fea)
            nbr_fea = torch.Tensor(nbr_fea)
            self_fea_idx = torch.LongTensor(self_fea_idx)
            nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

            if not self.disable_save_torch:
                with open(os.path.join(self.torch_data_path, cif_id+'.pkl'), 'wb') as f:
                    pickle.dump(
                        (atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx), f)

        return (atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx), target, cif_id
