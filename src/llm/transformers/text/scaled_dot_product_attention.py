import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention.
    """
    def __init__(self,  dropout=None):
        """
        :param d_k: Head dimension
        :param dropout: Dropout probability
        """
        super(ScaledDotProductAttention, self).__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        """
        Computes Scaled Dot-Product Attention.

        :param query: Query = [batch_size, n_heads, seq_len, d_k]
        :param key: Key = [batch_size, n_heads, seq_len, d_k]
        :param value: Value = [batch_size, n_heads, seq_len, d_k]
        :param mask: Mask = [batch_size, n_heads, seq_len, seq_len]
        :return: Output tensor and attention weights
        """
        
        # Calculate scaled dot product attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-1e9"))

        # Apply softmax to get attention probabilities
        attention_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Calculate attention-weighted sum of values
        output = torch.matmul(attention_weights, value)

        return output, attention_weights