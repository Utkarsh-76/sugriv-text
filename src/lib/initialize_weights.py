import torch.nn as nn

# should this be part of the model it self ?
def initialize_weights(model):
    """
    Weight initialization using Xavier uniform.
    """
    if hasattr(model, 'weights'):
        for weight in model.weights:
            if weight.dim() > 1:
                nn.init.xavier_uniform_(weight.data)