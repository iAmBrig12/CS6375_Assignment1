---

Requirements:

Install the following Python packages before running the scripts:

- torch
- scikit-learn
- tqdm
- argparse
- pickle
- json

Usage:

**Load Glove Embedding**

``python load_glove_embedding.py``  
Loads the GloVe embeddings and prepares them for use in the models.

**Preprocess Data**

``python data_preprocess.py``  
Preprocesses the input data, including tokenization and formatting for training.

**FFNN**

``python ffnn.py --hidden_dim 10 --epochs 1 ``
``--train_data ./training.json --val_data ./validation.json``  
Trains a Feedforward Neural Network (FFNN) with the specified parameters.

**RNN**

``python rnn.py --hidden_dim 32 --epochs 10 ``
``--train_data training.json --val_data validation.json``  
Trains a Recurrent Neural Network (RNN) with the specified parameters.

