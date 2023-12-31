"""
This lays out the structure of the model. The model is composed of the following components:
- Input Embeddings
- Positional Encoding
- Encoder
- Decoder
- Projection Layer
- Transformer
- Multi-Head Attention Block
- Feed Forward Block
- Encoder Block
- Decoder Block
- Residual Connection
- Layer Normalization
"""
import math
import torch
from torch import nn

class LayerNormalization(nn.Module):
    """
    Layer normalization is used instead of batch normalization because batch normalization
    is not effective for batch sizes of 1 or low values. 
    """
    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        """
        Forward pass of the layer normalization.
        :param x: the input tensor of shape (batch, seq_len, hidden_size) 
        """
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Feed forward block is a simple two layer neural network with a ReLU activation function.
    This is used to transform the output of the multi-head attention block to the input of the
    next multi-head attention block.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        """
        Forward pass of the feed forward block. 
        :param x: the input tensor of shape (batch, seq_len, d_model)
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    """
    Input embeddings are used to convert the input tokens into a vector of size d_model. 
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass of the input embeddings.
        :param x: the input tensor of shape (batch, seq_len)
        """
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Positional encoding is used to add the position of the token in the
    sentence to the embedding vector. 
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the positional encoding. 
        :param x: the input tensor of shape (batch, seq_len, d_model)
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)

        return self.dropout(x)

class ResidualConnection(nn.Module):
    """
    Residual connection is used to add the input of the sublayer to its output. 
    """
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """
        Forward pass of the residual connection.
        :param x: the input tensor of shape (batch, seq_len, d_model)
        """
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention block is used to calculate the attention between the input vectors.
    The input vectors are split into h heads, and the attention is calculated for each head. 
    We then combine the heads together to get the final output.
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Calculates the attention between the query and key vectors. The attention scores are used to
        calculate the weighted sum of the value vectors. 

        :param query: the query vector of shape (batch, h, seq_len, d_k)
        :param key: the key vector of shape (batch, h, seq_len, d_k)
        :param value: the value vector of shape (batch, h, seq_len, d_k)
        :param mask: the mask to apply to the attention scores
        :param dropout: the dropout layer to apply to the attention scores

        :return: the weighted sum of the value vectors and the attention scores, and the attention scores themselves
        """
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Apply softmax
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """
        Forward pass of the multi-head attention block. 

        :param q: the query vector of shape (batch, seq_len, d_model)
        :param k: the key vector of shape (batch, seq_len, d_model)
        :param v: the value vector of shape (batch, seq_len, d_model)
        :param mask: the mask to apply to the attention scores

        :return: the output of the multi-head attention block
        """
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

class EncoderBlock(nn.Module):
    """
    Encoder block is composed of two sublayers: multi-head attention block and feed forward block.
    The output of each sublayer is added to the input of the sublayer. We then normalize the output
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # We have two sublayers: multi-head attention block and feed forward block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        The forward pass of the encoder block.

        :param x: the input tensor of shape (batch, seq_len, d_model)
        :param src_mask: the mask to apply to the attention scores
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    """
    Encoder is composed of N encoder blocks. This is the part of the transformer that encodes the input, 
    and contains the input embeddings and positional encoding layers. 
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """
        Forward pass of the encoder.

        :param x: the input tensor of shape (batch, seq_len, d_model)
        :param mask: the mask to apply to the attention scores
        """
        for layer in self.layers:
            x = layer(x, mask)
        # Normalize the output
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    Decoder block is composed of three sublayers: multi-head attention block, cross attention block and feed forward block.
    The output of each sublayer is added to the input of the sublayer. We then normalize the output 
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        The forward pass of the decoder block. 

        :param x: the input tensor of shape (batch, seq_len, d_model)
        :param encoder_output: the output of the encoder of shape (batch, seq_len, d_model)
        :param src_mask: the mask to apply to the attention scores
        :param tgt_mask: the mask to apply to the attention scores
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    """
    The decoder is composed of N decoder blocks. This is the part of the transformer that decodes the input,
    and contains the input embeddings and positional encoding layers.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass of the decoder. 

        :param x: the input tensor of shape (batch, seq_len, d_model)
        :param encoder_output: the output of the encoder of shape (batch, seq_len, d_model)
        :param src_mask: the mask to apply to the attention scores
        :param tgt_mask: the mask to apply to the attention scores
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Projection layer is used to project the output of the decoder to the vocabulary size.
    """
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        A forward pass of the projection layer. Only need to apply a linear layer.

        :param x: the input tensor of shape (batch, seq_len, d_model)
        """
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)

class Transformer(nn.Module):
    """
    The transformer is composed of the encoder, decoder, input embeddings, positional encoding 
    layers and projection layer.  This is the main model that we will be using for training and 
    inference.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encode the source sentence.

        :param src: the source sentence of shape (batch, seq_len)
        :param src_mask: the mask to apply to the attention scores

        :return: the output of the encoder
        """
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Decode the transformer output.

        :param encoder_output: the output of the encoder of shape (batch, seq_len, d_model)
        :param src_mask: the mask to apply to the attention scores
        :param tgt: the target sentence of shape (batch, seq_len)
        :param tgt_mask: the mask to apply to the attention scores

        :return: the output of the decoder
        """
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Use the projection layer to project the output of the decoder to the vocabulary size.

        :param x: the input tensor of shape (batch, seq_len, d_model)
        """
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    """
    Build the transformer model based on the parameters. Most of the parameters are the same as the paper.

    :param src_vocab_size: the size of the source vocabulary
    :param tgt_vocab_size: the size of the target vocabulary
    :param src_seq_len: the maximum length of the source sentence
    :param tgt_seq_len: the maximum length of the target sentence
    :param d_model: the dimension of the model, 512 is the default mentioned in the paper
    :param N: the number of encoder and decoder blocks, 6 is the default mentioned in the paper
    :param h: the number of heads, 8 is the default mentioned in the paper
    :param dropout: the dropout rate, 0.1 is the default mentioned in the paper
    :param d_ff: the dimension of the feed forward block, 2048 is the default mentioned in the paper

    :return: the transformer model
    """
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters. This is important so that the model can learn effectively.
    # The parameters are initialized based on the method mentioned in the paper, Xavier uniform.
    # Xavier uniform is used to initialize the parameters to make sure the variance remains the same
    # in each layer. This is important because we are using residual connections and layer normalization.
    # If the variance is not the same, the model will not learn effectively.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
