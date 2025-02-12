import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from data import load_traindata
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split

batch_size = 32
num_subclasses = 200
learning_rate = 3e-4
epochs = 1000
test_epochs = 10
seq_size = 100
n_block = 4


# Encoder function (One-hot encoding)
def one_hot_encode(labels, unique_labels):
    # Create a mapping from unique labels to indices
    label_to_index = {label.item(): idx for idx, label in enumerate(unique_labels)}
    
    # Convert the labels to indices based on the mapping
    indices = torch.tensor([label_to_index[label.item()] for label in labels])
    
    # Create the one-hot encoded tensor
    return torch.eye(len(unique_labels))[indices]

# Decoder function (Converts one-hot back to original labels)
def one_hot_decode(one_hot, unique_labels):
    # Get the index of the '1' in the one-hot vector
    index = torch.argmax(one_hot)
    return torch.tensor(unique_labels[index])

#load dataset
X,Y = load_traindata(num_subclasses)
X = np.array(X)
X = torch.from_numpy(X)
X = X.view(num_subclasses, int(5000/seq_size),seq_size,12) #reshape after split
X = X.view(num_subclasses*(int(5000/seq_size)),seq_size,12)
unique_labels = Y  # Unique labels are just the 500 unique values in Y
Y = one_hot_encode(Y, unique_labels)  # One-hot encode Y
Y = Y.unsqueeze(1).repeat(1, int(5000/seq_size), 1).view(-1, num_subclasses)


class ECGDataset(Dataset):
    def __init__(self,X,Y):  
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return X[idx], Y[idx]
    

dataset = ECGDataset(X,Y)
train_size = int(0.9 * len(dataset)) 
test_size = len(dataset) - train_size  
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class RepresentationNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        block = nn.TransformerEncoderLayer(d_model=seq_size,nhead=5, dim_feedforward=768)
        self.transformer = nn.TransformerEncoder(block, num_layers=n_block)
    def forward(self,x):
        x = self.transformer(x)
        return x
class ECGRepresentation(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = RepresentationNetwork()
    def forward(self,x):
        B,T,C = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, seq_size)
        x = self.network(x)
        x = x.view(B, C, -1)
        x = x.mean(dim=1)
        return x
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg = ECGRepresentation()
        # Define the TransformerEncoderLayer blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=seq_size, nhead=5, dim_feedforward=768)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = 3)
        # Layer normalization
        self.fc1 = nn.Linear(seq_size, 128)
        self.relu = nn.ReLU()
        self.xnorm1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, num_subclasses)
    def forward(self, x):
        x = self.ecg(x)
        x = self.transformer(x)
        x = self.relu(self.fc1(x))
        x = self.xnorm1(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim = 1)
        return x


model = Model()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

losses = []

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for epoch in range(epochs):
    for x, y in dataloader:
        x,y = x.to(torch.float32), y.to(torch.float32)
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = m(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    if epoch % test_epochs == 0:
        print(f'Epoch {epoch}, loss: {loss.item()}')


torch.save(model,'./m-17_01')

import torch
# Assuming the model and test_loader have been defined
# model.eval() switches the model to evaluation mode
model.eval()
# Initialize variables to track correct predictions and total predictions
correct = 0
total = 0
# Disable gradient computation during evaluation
with torch.no_grad():
    # Loop over the test dataset
    for data, labels in test_loader:
        # Move data to the appropriate device (if using CUDA)
        data,labels = data.to(torch.float32), labels.to(torch.float32)
        data, labels = data.to(device), labels.to(device)
        # Get model predictions
        outputs = model(data)
        _, true_labels = torch.max(labels, 1)
        # Get the predicted class by taking the argmax (class with highest score)
        _, predicted = torch.max(outputs, 1)
        # Update the total number of samples and correct predictions
        total += labels.size(0)
        correct += (predicted == true_labels).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f'Accuracy on test dataset: {accuracy:.2f}%')
