import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import LabelEncoder

from utils.load_dataset import load_lsst_data
# from models.adapter import TimeSeriesFoundationClassifier


# Mock foundation model for demonstration purposes
class MockFoundationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=6, out_channels=768, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def main():
    # 1. Load and prepare the data
    print("Initializing data loading...")
    X_train, y_train, X_test, y_test = load_lsst_data()

    # Encode labels to integers (0 to num_classes - 1)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    num_classes = len(label_encoder.classes_)

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2. Initialize the Foundation Model and Adapter
    # Replace MockFoundationModel with your actual downloaded foundation model (e.g., MOMENT, Chronos, etc.)
    base_model = MockFoundationModel()

    model = TimeSeriesFoundationClassifier(
        foundation_model=base_model,
        embedding_dim=768,  # Ensure this matches your foundation model's output dimension
        num_classes=num_classes,
        freeze_base=True,
    )

    # 3. Setup Optimizer and Loss Function
    criterion = nn.CrossEntropyLoss()
    # Since we froze the base model, we only pass the classifier parameters to the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

    # 4. Training Loop (Example: 5 Epochs)
    model.train()
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader):.4f}")


if __name__ == "__main__":
    main()
