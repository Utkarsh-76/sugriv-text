import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    This class is designed for generating positions encodings.
    The PositionalEncoding class will inherit the nn.Module class.

    """

    def __init__(self, d_model, max_sequence_length):
        """
        This is the constructor method for the positional encoding
        class. It initializes the object when an instance of the
        class is created.

        Parameters
        ----------
        d_model : intger
            dimension of the positional encoding.
        max_sequence_length : integer
            maximum sequence length

        """

        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self,x):

        """
        This method calculates and returns the positional encodings
        as per the mathematical formulation.


        Returns
        -------
        PE: tensor
            tensor of dimensions d_model x max_sequence_length

        """
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(x.shape[1]).float().unsqueeze(1)
        even_position_encoding = torch.sin(position / denominator)
        odd_position_encoding = torch.cos(position / denominator)
        stacked = torch.stack([even_position_encoding, odd_position_encoding], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        PE = PE.unsqueeze(0).expand(x.size(0), -1, -1)  # Expand PE to match batch size
        return PE