# Lite Observation
[arXiv paper](https://arxiv.org/abs/1706.03762)

A light implementation of the 2017 Google paper 'Attention is all you need'.

For this implementation I will implement a translation from English to Spanish, as Tranformer models are exceptional at language 
translation and this seems to be a common use of light implementations of this paper.

The dataset I will be using is the [opus books](https://opus.nlpl.eu/Books.php) dataset which is a collection of copyright free books.
The book content of these translations are free for personal, educational, and research use. 
[OPUS language resource paper](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf).

## Notes
I'm creating notes as I go, which can be found in [NOTES.md](./NOTES.md).

## Transformer model architecture
![Transformer model](./resources/transformer-model.png)


## Requirements
- PyTorch
> `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
Since I have an AMD card, I am going to use CPU training. However since AMD RocM will allow for using CUDA on
and AMD card, I hope to use that to get GPU training by the end of the block.

## TODO and tenative timeline:
- [X] Input Embeddings
- [X] Positional Encoding
- [X] Layer Normalization **- Due by 11/1**
- [X] Feed forward
- [X] Multi-Head attention
- [X] Residual Connection
- [X] Encoder
- [X] Decoder **- Due by 11/8**
- [X] Linear Layer
- [X] Transformer
- [X] Tokenizer **- Due by 11/15**
- [X] Dataset
- [X] Training loop
- [ ] Visualization of the model **- Due by 11/22**
- [X] Install AMD RocM to train with GPU **- Attempt to do by end**

## References used
- [Dropout information](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)
- [Input embedding and positional encoding video](https://www.youtube.com/watch?v=3mTsYm9qQFA)
- [arXiv paper](https://arxiv.org/abs/1706.03762)
- [Transformer model overview](https://www.youtube.com/watch?v=4Bdc55j80l8)
