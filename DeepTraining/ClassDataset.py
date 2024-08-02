import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import shap 

class EmbeddingsDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embedding_path = self.file_paths[idx]  # Get the path of the embedding
        embedding = np.load(embedding_path)

        label = self.labels[idx]

        return {
            'embedding': torch.tensor(embedding, dtype=torch.float32).squeeze(0), 
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define file paths
file_dir1 = '/homes/nva01/EmbeddingsNaive'
file_dir2 = '/homes/nva01/EmbeddingsNaiveT'

# List all .npy files from both directories
file_paths1 = [os.path.join(file_dir1, file) for file in os.listdir(file_dir1) if file.endswith('.npy')]
file_paths2 = [os.path.join(file_dir2, file) for file in os.listdir(file_dir2) if file.endswith('.npy')]

# Combine all file paths
all_file_paths = file_paths1 + file_paths2
print(all_file_paths)

# CSV with embeddings
csv_file = '/homes/nva01/DeepTraining/Csvs/TotalEmbeddingsNaive.csv'

# Ensure the CSV file path is correctly passed
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"The file {csv_file} does not exist")

# Reading the embeddings CSV
df = pd.read_csv(csv_file)

# Checking the unique labels
unique_labels = df['label'].unique()
num_classes = len(unique_labels)
label_to_int = {label: i for i, label in enumerate(sorted(unique_labels))}
print(f"Number of unique classes: {num_classes}")

# Create a list of labels for each file path
labels = []
for file_path in all_file_paths:
    filename = os.path.basename(file_path)
    label = df[df['filename'] == filename]['label'].values[0]
    labels.append(label_to_int[label])

# Create a dataset
dataset = EmbeddingsDataset(all_file_paths, labels)

# Split into train, val, and test datasets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Building a classifier model with the embeddings already extracted by the Wav2Vec2 model
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Create the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Classifier(num_classes=num_classes)
model.to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)

# Training the model
n_epochs = 350
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
all_preds = []
all_labels = []

for epoch in range(n_epochs):
    train_loss = 0.0
    val_loss = 0.0
    correct_train = 0
    total_train = 0
    correct_val = 0
    total_val = 0

    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs = data['embedding'].to(device)
        labels = data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs = data['embedding'].to(device)
            labels = data['label'].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss / len(val_loader))
    
    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss / len(train_loader):.3f}, Val Loss: {val_loss / len(val_loader):.3f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

# Graphs of loss and validation
# Plotting Loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot = Loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting Accuracy
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot = Accuracy
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_loss_accuracyNaive.png')

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(20, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrixNaive.png')

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(20, 7))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Normalized Confusion Matrix')
plt.savefig('normalized_confusion_matrixNaive.png')

subset_indices = np.random.choice(len(dataset), 100, replace=False)
subset = torch.utils.data.Subset(dataset, subset_indices)
subset_loader = DataLoader(subset, batch_size=len(subset), shuffle=False)

# Extract embeddings/features for the subset
for data_batch in subset_loader:
    subset_embeddings = data_batch['embedding'].to(device)
    subset_labels = data_batch['label'].to(device)
    break  # Only need one batch since batch_size = len(subset)

# Initialize SHAP DeepExplainer with a smaller background dataset
background = subset_embeddings[:10]  # Adjust as needed

def predict_fn(x):
    return model(x).detach().cpu().numpy()

background = background.to(device)

# Initialize SHAP GradientExplainer
explainer = shap.GradientExplainer(model, background)

# Compute SHAP values
shap_values = explainer.shap_values(subset_embeddings)

# Ensure SHAP values are properly shaped
if isinstance(shap_values, list):
    shap_values_combined = np.mean(shap_values, axis=0)  # Combine SHAP values across classes
else:
    shap_values_combined = shap_values

subset_embeddings_np = subset_embeddings.cpu().numpy()

# Plot the aggregated SHAP values
plt.figure()
shap.summary_plot(shap_values_combined, subset_embeddings_np, show=False)
plt.savefig('shap_summary_plot_combined.png')