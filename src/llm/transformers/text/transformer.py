import torch
import torch.nn as nn
import math
from src.llm.transformers.text.decoder_layer import DecoderLayer
from src.llm.transformers.text.positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    """
    Decoder-Only Transformer.
    """
    def __init__(self, pad_index, vocab_size, d_model, max_seq_len, n_heads, n_layers, ff_dim, dropout_rate,batch_size):

        """
        Initialize a Transformer decoder with N layers. Decoder self-attention layers are masked
        so that an attention head cannot attend to future words during training.

        :param pad_index: Index of the padding token '[PAD]'
        :param vocab_size: The size of the vocabulary
        :param d_model: The size of hidden dimensions
        :param max_seq_len: Maximum total input sequence length after tokenization
        :param n_heads: Number of heads in the multi-headed attention for each layer
        :param n_layers: Number of decoder layers
        :param ff_dim: Position-wise feed-forward size
        :param dropout: Dropout probability
        """
        super(Transformer, self).__init__()

        self.pad_index = pad_index
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.batch_size =  batch_size
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model,n_heads,ff_dim,dropout_rate) for _ in range(n_layers)])
        self.fc_linear = nn.Linear(int(d_model), int(vocab_size))

    def forward(self, input):
        """
        Decoder-Only Transformer forward pass.

        :param input: Input = [batch_size, seq_len]
        :return: Output = [batch_size, seq_len, vocab_size] and
                 attention = [batch_size, n_heads, seq_len, seq_len]
        """

        # convert to long
        input = input.long()

        # multiplication to reduce variance in the embeddings
        output = input * math.sqrt(self.d_model)

        # output = [batch_size, seq_len, d_model]
        output = self.pos_enc(output)
        
        # create the attention mask ( batch_size X d_model )
        attention_mask = self.create_causal_mask(self.batch_size, self.d_model)

        # apply attention to positional encoded output at each layer
        for layer in self.layers:
            output, attention = layer(output, attention_mask)

        # apply the linear layer to the output
        output = self.fc_linear(output)

        # get the output and attention
        return output, attention


    def create_causal_mask(self,batch_size, seq_length):
        """
        Create a causal mask to prevent attending to future positions.

        Args:
            batch_size (int): The batch size.
            seq_length (int): The sequence length.

        Returns:
            torch.Tensor: A 3D tensor of shape (batch_size, seq_length, seq_length),
                where mask[i, j, k] is 1 if k <= j, and 0 otherwise.
        """
        # Create a boolean mask where True represents positions that should be masked
         # Create a lower triangular matrix of shape (seq_length, seq_length)
        mask = torch.tril(torch.ones((seq_length, seq_length), dtype=torch.bool))

        # Expand the mask to match the batch size
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        return mask
