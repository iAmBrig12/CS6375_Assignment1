import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
from torch.utils.data import DataLoader, TensorDataset

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)  # Added dim=1
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        input_vector = input_vector.to(device)  # Move input to device
        predicted_label = self.W1(input_vector)
        predicted_label = self.activation(predicted_label)
        predicted_vector = self.W2(predicted_label)
        predicted_vector = self.softmax(predicted_vector)
        return predicted_vector

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

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
        tra.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"] - 1)))
    for elt in test:
        tes.append((elt["text"].split(), int(elt["stars"] - 1)))
    return tra, val, tes

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training") #added batch size
    args = parser.parse_args()

    random.seed(69)
    torch.manual_seed(69)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    test_data = convert_to_vector_representation(test_data, word2index)

    train_vectors = torch.stack([ex[0] for ex in train_data]).to(device)
    train_labels = torch.tensor([ex[1] for ex in train_data]).to(device)
    valid_vectors = torch.stack([ex[0] for ex in valid_data]).to(device)
    valid_labels = torch.tensor([ex[1] for ex in valid_data]).to(device)
    test_vectors = torch.stack([ex[0] for ex in test_data]).to(device)
    test_labels = torch.tensor([ex[1] for ex in test_data]).to(device)

    train_dataset = TensorDataset(train_vectors, train_labels)
    valid_dataset = TensorDataset(valid_vectors, valid_labels)
    test_dataset = TensorDataset(test_vectors, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print(f"========== Training for {args.epochs} epochs ==========")
    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        print(f"Training started for epoch {epoch + 1}")
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.compute_Loss(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Training completed for epoch {epoch + 1}")
        print(f"Training accuracy for epoch {epoch + 1}: {correct / total}")
        print(f"Training time for this epoch: {time.time() - start_time}")

        model.eval()
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Validation started for epoch {epoch + 1}")

        with torch.no_grad():
            for inputs, labels in tqdm(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to device
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation completed for epoch {epoch + 1}")
        print(f"Validation accuracy for epoch {epoch + 1}: {correct / total}")
        print(f"Validation time for this epoch: {time.time() - start_time}")

    print("========== Training completed ==========")
    print("========== Testing started ==========")
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for inputs, labels in tqdm(test_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Testing accuracy: {correct / total}")
    print("========== Testing completed ==========")