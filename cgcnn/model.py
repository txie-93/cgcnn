import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_add


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len, enable_tanh=False):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        if enable_tanh:
            self.act1 = nn.Tanh()
            self.act2 = nn.Tanh()
        else:
            self.act1 = nn.Softplus()
            self.act2 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.pooling = SumPooling()

    def forward(self, atom_in_fea, nbr_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        atom_self_fea = atom_in_fea[self_fea_idx, :]

        total_fea = torch.cat([atom_self_fea, atom_nbr_fea, nbr_fea], dim=1)

        total_fea = self.fc_full(total_fea)
        total_fea = self.bn1(total_fea)

        filter_fea, core_fea = total_fea.chunk(2, dim=1)
        filter_fea = self.sigmoid(filter_fea)
        core_fea = self.act1(core_fea)

        # take the elementwise product of the filter and core
        nbr_msg = filter_fea * core_fea
        nbr_sumed = self.pooling(nbr_msg, self_fea_idx)

        nbr_sumed = self.bn2(nbr_sumed)
        out = self.act2(atom_in_fea + nbr_sumed)

        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, enable_tanh=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        classification: bool
          If classification should be done instead of regression
        enable_tanh: bool
          Use tanh instead of softplus
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len, enable_tanh=enable_tanh)
                                    for _ in range(n_conv)])
        self.pooling = MeanPooling()
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_fea)

        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx)

        crys_fea = self.pooling(atom_fea, crystal_atom_idx)

        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))

        out = self.fc_out(crys_fea)

        if self.classification:
            out = self.logsoftmax(out)
        return out


class MeanPooling(nn.Module):
    """
    mean pooling
    """

    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, x, index):

        mean = scatter_mean(x, index, dim=0)

        return mean

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class SumPooling(nn.Module):
    """
    mean pooling
    """

    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, x, index):

        mean = scatter_add(x, index, dim=0)

        return mean

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)
