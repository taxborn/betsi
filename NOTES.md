# Input embedding
![Input embedding example](./resources/input-embedding.png)
This is discussed in section 3.4 in the arXiv paper. Input embeddings is a way to represent our input in a vector of `n` dimensions (in the image, 512 dims are used). 
We do this by first translating our input sentence (or list of tokens) into their input id's which translates to
an embedding, which is a vector of size `n`. This vector is learned by the model.

## Positional encoding
![Positional encoding example](./resources/positional-encoding.png)
In addition to input embedding, we have an additive encoding 
