import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# Load CSV file
df = pd.read_csv("vasicek_results.csv")

# Convert DataFrame to NumPy arrays
X = df.iloc[:, :-3].values  # All columns except last 3 (input features)
y = df.iloc[:, -3:].values  # Last 3 columns (output labels)

# Standardize the features before creating dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PyTorch Dataset
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CustomDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Get input feature size dynamically
input_size = X.shape[1]

# Model with Xavier Initialization
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 3)
        self.relu = nn.ReLU()
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize Model
model = Module()

# Loss & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Lower learning rate

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Load Test Data
data_test = pd.read_csv("vasicek_test.csv")
X_test = data_test.iloc[:, :-3].values
y_test = data_test.iloc[:, -3:].values

# Standardize Test Data Before Creating Dataset
X_test = scaler.transform(X_test)
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate Model
model.eval()
test_loss = 0
num_samples = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        batch_size = batch_X.shape[0]
        test_loss += loss.item() * batch_size
        num_samples += batch_size

        print("outputs: ", outputs[0])
        print("batch_y: ", batch_y[0])

test_loss /= num_samples  # True average test loss
print(f"Test Loss: {test_loss:.4f}")
