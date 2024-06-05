import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Projects to ff_dim and then back to d_model.
    """
    def __init__(self, d_model, ff_dim, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.

        :param d_model: Dimensionality of input and output
        :param ff_dim: Dimensionality of the inner layer
        :param dropout: Dropout probability
        """
        super(PositionwiseFeedForwardNetwork, self).__init__()

        self.linear_1 = nn.Linear(d_model, ff_dim)
        self.linear_2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for position-wise FFN

        :param x: Input = [batch_size, seq_len, d_model]
        :return: Output = [batch_size, seq_len, d_model]
        """
        # project input to [batch_size, seq_len, ff_dim]
        x = F.relu(self.linear_1(x))

        x = self.dropout(x)

        x = self.linear_2(x)

        return x
