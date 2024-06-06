import torch.nn as nn
from src.llm.transformers.text.masked_multiheaded_attention import MaskedMultiHeadAttention
from src.llm.transformers.text.positional_feed_forward_network import PositionwiseFeedForwardNetwork

class DecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    """
    def __init__(self, d_model=768, n_heads=1, ff_dim=4, dropout=0.1):
        """
        A single Transformer decoder layer made of masked self-attention and feed-forward.

        :param d_model: Size of the model (size of each vector)
        :param n_heads: Number of heads (n_heads)
        :param ff_dim: Dimensionality of feed-forward inner layer (seq_len)
        :param dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()

        self.masked_self_attention = MaskedMultiHeadAttention(d_model, n_heads, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.attention_layer_norm = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForwardNetwork(d_model, ff_dim)
        self.dropout_2 = nn.Dropout(dropout)
        self.ffn_layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, mask):
        """
        Forward pass of Transformer decoder layer.

        :param input: Input = [batch_size, seq_len, d_model]
        :param mask: Mask = [batch_size, seq_len, seq_len]
        :return: Output = [batch_size, seq_len, d_model] and
                 attention weights = [batch_size, n_heads, seq_len, seq_len]
        """
        # decoder masked self-attention
        output, attention_weights = self.masked_self_attention(input, input, input, mask)
        output = self.dropout_1(output)

        # residual connection and layer norm
        output = self.attention_layer_norm(input + output)

        # feed forward layer
        ffn_output = self.ffn(output)
        ffn_output = self.dropout_2(ffn_output)

        # residual connection and layer norm
        ffn_output = self.ffn_layer_norm(output + ffn_output)
        
        # return output and attention weights
        return ffn_output, attention_weights