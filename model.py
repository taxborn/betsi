import math
import torch
import torch.nn as nn

"""
Input Embeddings
"""
class InputEmbeddings(nn.Module):
    def __init__(self, dimensions: int, vocabulary: int):
        super().__init__()
        self.dimensions = dimensions  # d_model in the paper
        self.vocabulary = vocabulary
        self.embedding = nn.Embedding(vocabulary, dimensions)

    def forward(self, x):
        # Page 5: Multiply the weights by sqrt(dimensions)
        return self.embedding(x) * math.sqrt(self.dimensions)

"""
Positional Encoding. Added to input embedding.
"""
class PositionalEncoding(nn.Module):
    # We need sequence_length which is the amount of tokens in the input.
    def __init__(self, dimensions: int, sequence_length: int, dropout: float):
        super().__init__()
        self.dimensions = dimensions  # d_model in the paper
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (sequence_length, dimensions)
        positional_encoding = torch.zeros(sequence_length, dimensions)
        # Create a vector/tensor of shape (sequence_length, 1)
        # arange = 1D tensor (vector)
        positions = torch.arange(0, length, dtype=torch.float).unsqueeze(1)

        denominator = torch.exp(torch.arange(0, dimensions, 2).float() * (-math.log(10000.0) / dimensions))

        # Apply the sin to even positions
        positional_encoding[:,0::2] = torch.sin(positions * denominator)
        # Apply the cos to odd positions
        positional_encoding[:,1::2] = torch.cos(positions * denominator)

        # We are now at size (1, sequence_length, dimensions)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        # This will mean it is not a learned tensor
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
