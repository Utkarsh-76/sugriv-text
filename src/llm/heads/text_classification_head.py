import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.llm.transformers.text.transformer import Transformer

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

NUM_CLASSES = int(os.getenv("NUM_CLASSES"))

class TextClassificationModel(nn.Module):

    def __init__(self, pad_index, d_model, max_seq_length, n_heads, n_layers, ff_dim, dropout_rate, vocab_size, batch_size):
        super(TextClassificationModel, self).__init__()

        # the base model
        self.base_model = Transformer(pad_index, vocab_size, d_model, max_seq_length, n_heads, n_layers, ff_dim, dropout_rate, batch_size)

        # create a linear layer
        self.linear = nn.Linear( int(d_model),int(NUM_CLASSES))  # Input size: 768, Output size: 2

    def forward(self, input_ids=None, attention_mask=None, pipeline=None, labels=None):

        # Get the outputs from the transformer
        outputs = self.base_model(input_ids)[0]
        
        # Use the last token representation for classification
        last_token_output = outputs.last_hidden_state[:, -1, :]
        
        # Pass the last token representation through leniar layer
        logits = self.linear(last_token_output)

        # Calculate the loss if labels are provided
        loss = None
        if labels is not None:
            # Use cross-entropy loss for multi-class classification
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return logits, loss