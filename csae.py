# %%
import time

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import transforms
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
# %%
# Define a data transformation to convert images to tensors
transform = transforms.ToTensor()

# Load the MNIST dataset for training and validation
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=False, transform=transform
)
valid_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=False, transform=transform
)
# Create a data loader for training data with a batch size of 100
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=100)

# Check for GPU availability and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# %%
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class CSAE(nn.Module):
    def __init__(self):
        super(CSAE, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 24 * 24, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.dec = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 24 * 24),
            nn.ReLU(),
            Reshape(-1, 64, 24, 24),
            nn.ConvTranspose2d(64, 32, 3, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 1),
            nn.ReLU(),
        )
        
        self.clf = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
#             nn.Softmax(dim=1),
        )

    def forward(self, x):
        y = self.enc(x)
        output = self.dec(y)
        
        output2 = self.clf(y)
        return output, output2
# %%
model = CSAE().to(device)
# %%
ae_loss = nn.MSELoss()
clf_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
# %%
train_loss_ae = []
train_loss_clf = []
num_epochs = 100

start_time = time.perf_counter()

iter = 0
for epoch in tqdm(range(num_epochs)):
    train_epoch_loss_ae = 0
    train_epoch_loss_clf = 0
    # Iterate through batches of training data
    for imgs, labels in train_dl:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        output, output2 = model(imgs)
        loss1 = ae_loss(output, imgs)
        train_epoch_loss_ae += loss1.cpu().detach().numpy()
        loss1.backward(retain_graph=True)
        optimizer.step()
        
        loss2 = clf_loss(output2, labels)
        train_epoch_loss_clf += loss2.cpu().detach().numpy()
        loss2.backward()
        optimizer.step()
    
    train_loss_ae.append(train_epoch_loss_ae)
    train_loss_clf.append(train_epoch_loss_clf)

print(f"{(time.perf_counter() - start_time):.4f} seconds elapsed!")
# 840.4907 seconds elapsed!
# %%
plt.plot(train_loss_ae)
# %%
plt.plot(train_loss_clf)
# %%
# Initialize variables to store latent representations and labels
values = None
all_labels = []
# Generate latent representations for the entire training dataset
with torch.no_grad():
    for imgs, labels in train_dl:
        imgs = imgs.to(device)
        all_labels.extend(list(labels.numpy()))
        latents = model.enc(imgs)
        if values is None:
            values = latents.cpu()
        else:
            values = torch.vstack([values, latents.cpu()])
# %%
# Create a color map for visualization
cmap = plt.get_cmap("viridis", 10)

# Plot the scatter plot of latent space with color-coded labels
all_labels = np.array(all_labels)
values = values.numpy()
pc = plt.scatter(values[:, 0], values[:, 1], c=all_labels, cmap=cmap)
plt.colorbar(pc)
# %%
