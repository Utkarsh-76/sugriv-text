import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MaskedMultiHeadAttention(nn.Module):
    """ Masked Multihead attention layer """
    def __init__(self, d_model, num_heads, dropout=0.1):
        """ Applies the scaled dot product attention in parallel for K,Q,V pairs """
        super(MaskedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_fc = nn.Linear(d_model, d_model)
        self.key_fc = nn.Linear(d_model, d_model)
        self.value_fc = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """ Apply the mask to the K,Q,V pairs and then calculate 
            the output attention weights"""
        
        batch_size = query.size(0)

        # Linear transformation
        Q = self.query_fc(query)
        K = self.key_fc(key)
        V = self.value_fc(value)

        # Reshape for multi-heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, query_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, key_len, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, value_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, query_len, key_len)

        scores = scores.masked_fill(scores == 0, -np.inf)

        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, query_len, key_len)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Compute weighted sum
        weighted_sum = torch.matmul(attention_weights, V)  # (batch_size, num_heads, query_len, head_dim)

        # Concatenate heads and linear transformation 
        weighted_sum = weighted_sum.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, query_len, d_model)

        output = self.fc_out(weighted_sum)  # (batch_size, query_len, d_model)

        return output, attention_weights
