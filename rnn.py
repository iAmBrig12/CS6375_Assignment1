import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 8
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        inputs = inputs.to(device)  # Move input to device
        x, _ = self.rnn(inputs)
        x = torch.sum(x, dim=0)
        x = self.W(x)
        x = self.softmax(x)

        return x


def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = []
    val = []
    tes = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in test:
        tes.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val, tes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)  # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim).to(device)  # Move model to GPU if available
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    def get_word_vector(word):
        return word_embedding.get(word.lower(), word_embedding['unk'])

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    while epoch < args.epochs:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [get_word_vector(i) for i in input_words]

                # Transform the input into required shape and move to device
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)
                gold_label = torch.tensor([gold_label]).to(device)  # Move label to device
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), gold_label)

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total

        
        print("Validation started for epoch {}".format(epoch + 1))
        with torch.no_grad():
            model.eval()
            correct = 0
            total = 0
            random.shuffle(valid_data)
            valid_data = valid_data

            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [get_word_vector(i) for i in input_words]

                vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)  # Move to device
                gold_label = torch.tensor([gold_label]).to(device)  # Move label to device
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            validation_accuracy = correct/total

        epoch += 1


    print("========== Training completed ==========")
    print("========== Testing started ==========")
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        random.shuffle(test_data)
        test_data = test_data

        for input_words, gold_label in tqdm(test_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [get_word_vector(i) for i in input_words]
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1).to(device)  # Move to device
            gold_label = torch.tensor([gold_label]).to(device)  # Move label to device
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
        test_accuracy = correct / total
        print("Testing accuracy: {}".format(test_accuracy))


    with open("rnn_results.out", "a") as f:
        f.write("-"*60 + "\n")
        f.write(f"Args: {args}\n")
        f.write(f"Training accuracy: {last_train_accuracy}\n")
        f.write(f"Validation accuracy: {last_validation_accuracy}\n")
        f.write(f"Testing accuracy: {test_accuracy}\n")
