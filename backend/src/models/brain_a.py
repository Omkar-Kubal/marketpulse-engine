
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

class BrainADataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 30):
        """
        Args:
            X: Feature matrix (samples, features)
            y: Target vector (samples,)
            seq_len: Length of sequence for LSTM
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        # Sequence of length seq_len
        x_seq = self.X[idx : idx + self.seq_len]
        # Target associated with the END of the sequence (or the step after)
        # Assuming y is aligned with X. If y[i] is target for X[i], then for sequence ending at i, we want y[i].
        # But wait, y[i] is usually "return next period".
        # So for sequence X[0..29], we predict y[29] (which is return from 29->30).
        y_label = self.y[idx + self.seq_len - 1]
        
        return x_seq, y_label

class BrainA(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(BrainA, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Batch Norm for input
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention Mechanism (Optional, but good for "intelligence")
        # Simple attention: 
        # let's stick to last hidden state first for robustness
        
        # Fully Connected Output Heads
        self.fc_direction = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid() # Probability of Buy/Up
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # BN expects (batch, features, seq_len) for 1d, but that's for temporal convolution.
        # For sequence, standard manual norm or LayerNorm is better.
        # Let's skip BN on input or apply per step.
        # Actually LayerNorm is better for RNNs.
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0)) 
        # out: (batch, seq_len, hidden_dim)
        # hn: (num_layers, batch, hidden_dim)
        
        # Use last hidden state
        embedding = out[:, -1, :] # (batch, hidden_dim)
        
        # Heads
        direction_prob = self.fc_direction(embedding)
        
        return direction_prob, embedding

def train_brain_a(X: np.ndarray, y: np.ndarray, 
                  input_dim: int,
                  seq_len: int = 30,
                  epochs: int = 50,
                  batch_size: int = 32,
                  lr: float = 0.001) -> Tuple[BrainA, Dict]:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training Brain A on {device}")
    
    dataset = BrainADataset(X, y, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = BrainA(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    metrics = {
        'loss': [],
        'accuracy': []
    }
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
        avg_loss = epoch_loss / len(dataloader)
        acc = correct / total
        
        metrics['loss'].append(avg_loss)
        metrics['accuracy'].append(acc)
        
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
            
    return model, metrics
