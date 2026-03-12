import torch
import torch.nn as nn

class TimeSeriesFoundationClassifier(nn.Module):
    def __init__(self, foundation_model, embedding_dim, num_classes, freeze_base=True):
        """
        Adapts a foundation model for time series classification.
        """
        super().__init__()
        self.foundation_model = foundation_model
        
        # Freeze the base model to only train the classifier head (Linear Probing)
        if freeze_base:
            for param in self.foundation_model.parameters():
                param.requires_grad = False
                
        # The classification head
        self.classifier = nn.Sequential(
            # Assuming the foundation model outputs (batch, embedding_dim, timesteps)
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        # Extract features using the foundation model
        features = self.foundation_model(x)
        
        # Pass through the classification head
        logits = self.classifier(features)
        return logits