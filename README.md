# Lite Observation
[arXiv paper](https://arxiv.org/abs/1706.03762)
![Transformer model](./resources/transformer-model.png)

A light implementation of the 2017 Google paper 'Attention is all you need'.

For this implementation I will implement a translation from English to Spanish, as Tranformer models are exceptional at language 
translation and this seems to be a common use of light implementations of this paper.

The dataset I will be using is the [opus books](https://opus.nlpl.eu/Books.php) dataset which is a collection of copyright free books.
The book content of these translations are free for personal, educational, and research use. 
[OPUS language resource paper](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf).

## Notes
I'm creating notes as I go, which can be found in [NOTES.md](./NOTES.md).

## Requirements
- PyTorch
> `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
Since I have an AMD card, I am going to use CPU training. However since AMD RocM will allow for using CUDA on
and AMD card, I hope to use that to get GPU training by the end of the block.

## TODO:
- [X] Input Embeddings
- [X] Positional Encoding
- [ ] Layer Normalization
- [ ] Feed forward
- [ ] Multi-Head attention
- [ ] Residual Connection
- [ ] Encoder
- [ ] Decoder
- [ ] Linear Layer
- [ ] Transformer
- [ ] Tokenizer
- [ ] Dataset
- [ ] Training loop
- [ ] Visualization of the model
- [ ] Install AMD RocM to train with GPU
