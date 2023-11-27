# What I want to go over with Rushit
- This is a public repository that has the weights
- Show notes that go over each section
- Show code
    - **model.py** is the main model, builds up each section of the model
    - **train.py** is the training code to train the model.
        - This can pick up where it left off if it ends in the middle of an epoch, or just continue off of an old model
        - **train.out** is the resulting command line output during the training process, shows 2 examples after each epoch, shows how it gets better over time
    - **dataset.py** is used to construct a class for accessing the dataset
    - **config.py** is used to configure the model
    - **dna.py** is just a helper to print out the weights to human-readable text of the models
    - **translate.py** is used to actually use the model to translate text
    - **tokenizer_\*.json** is used to store the ID for the vocabularies
    - **attention_visualization.ipynb** Notebook going through some visualizations. I did not write this code, and was taken from another repository working on transformer models
- Show what training looks like with validation at each step on
- Show translate.py on a given sentence. I do not guarentee that this works great, just that it does what I am saying it does

# The model and weights
Trained on an AMD Radeon 7900 XT, 20GB of vRAM for 40 epochs, taking around 8 hours.
Follows the exact same parameters as in the paper for dimensions of matricies, so working with just about the exact same model.
Trained on an open-source dataset translating English to Italian.

# Where to improve
I went with Italian because of a smaller dataset, Spanish has more data (about twice the size), but takes twice as long to train.
Going with larger datasets will give me a larger vocabulary to choose from, and to give more context to words. Currently, en en-it,
we only have 15k English words, where there is really close to a million?

Training time is roughly linear with dataset size, but would need to experiment more to confirm that and to compare if that is 
consistent with other transformer models.

If I give it more training time and/or more memory (using an AIMUX lab computer), 

There are many, many papers since this paper exploring modifications to the model which improves performance and accuracy. One of which is 
BEAM decoding, which allows for faster lookups and longer decoding, I could look into implementing this.

There are scoring algorithms like BLEU which is specific for translation tasks, look into how I could implement that evaluation rather than just
translating by hand and loss.

There are some perfect matches during training, so the resulting model may be a bit overfit. I could adjust the learning rate and other parameters, 
and need to work on increasing the dataset size