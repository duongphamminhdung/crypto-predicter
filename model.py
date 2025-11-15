import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CryptoPredicter(nn.Module)                                                     : 
    def __init__(self, input_size=22, hidden_layer_size=100, num_layers=2, dropout=0.2): 
        super().__init__()
        # 1. Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Model configured to use device: {self.device}")
        print(f"🧠 Model input size: {input_size} features (technical indicators)")

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, dropout=dropout, batch_first=True)
        
            # Attention mechanism - learns to focus on important timesteps with bias toward recent
        self.attention = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, 1)
        )
        
                          # FC layer before output heads for better representation
        self.fc_hidden = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.signal_head = nn.Linear(hidden_layer_size, 2)  # 0: sell, 1: buy
        self.tp_head = nn.Linear(hidden_layer_size, 1)
        self.sl_head = nn.Linear(hidden_layer_size, 1)

        # 2. Move model to the selected device
        self.to(self.device)
        
        # Time decay factor for attention (higher = more emphasis on recent)
        self.time_decay_alpha = 3.0

    def forward(self, input_seq):
        # Get LSTM outputs for all timesteps
        lstm_out, _ = self.lstm(input_seq)  # Shape: (batch, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = lstm_out.shape
        
        # Calculate attention scores for each timestep
        # Shape: (batch, seq_len, 1)
        attention_scores = self.attention(lstm_out)
        
        # Add time-decay bias to favor more recent timesteps
        # Create position weights: more recent = higher weight
        positions = torch.linspace(0, 1, seq_len, device=self.device)  # 0 (oldest) to 1 (most recent)
        time_bias = torch.exp(self.time_decay_alpha * (positions - 1))  # Exponential decay
        time_bias = time_bias.unsqueeze(0).unsqueeze(-1)  # Shape: (1, seq_len, 1)
        
        # Combine learned attention with time decay bias
        attention_scores = attention_scores + time_bias
        
        # Apply softmax to get attention weights (sum to 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: (batch, seq_len, 1)
        
        # Weighted sum of LSTM outputs using attention weights
        # Shape: (batch, seq_len, hidden_size) * (batch, seq_len, 1) -> (batch, hidden_size)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)  # Shape: (batch, hidden_size)
        
        # Pass through FC layer for better representation
        fc_out = F.relu(self.fc_hidden(attended_output))
        fc_out = self.dropout_layer(fc_out)
        
        # Generate predictions from attended representation
        signal = self.signal_head(fc_out)
        take_profit = self.tp_head(fc_out)
        stop_loss = self.sl_head(fc_out)
        
        return F.softmax(signal, dim=1), take_profit, stop_loss

    def train_model(self, X_train, y_train_signal, y_train_tp, y_train_sl, epochs=10, lr=0.001, time_weighted=True):
        """
        Train the model with optional time-weighted loss.
        
        Args:
            time_weighted: If True, recent samples get higher weight (exponential decay from most recent)
        """
        signal_loss_func = nn.CrossEntropyLoss(reduction='none')  # Changed to 'none' for manual weighting
        price_loss_func  = nn.MSELoss(reduction='none')
        optimizer        = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Calculate time weights if enabled
        if time_weighted:
            # Exponential decay: more recent = higher weight
            # weights go from e^(-alpha) to 1.0 (most recent)
            alpha = 3.0  # Decay rate (higher = more emphasis on recent)
            positions = torch.linspace(0, 1, len(X_train))  # 0 (oldest) to 1 (newest)
            time_weights = torch.exp(alpha * (positions - 1))  # Peaks at 1.0 for most recent
            time_weights = time_weights / time_weights.sum() * len(X_train)  # Normalize
            time_weights = time_weights.to(self.device)
            print(f"⏰ Time-weighted training enabled (alpha={alpha})")
            print(f"   Weight range: {time_weights.min():.3f} (oldest) → {time_weights.max():.3f} (newest)")
        else:
            time_weights = torch.ones(len(X_train)).to(self.device)

        for i in range(epochs):
            total_loss = 0
            for j in range(len(X_train)):
                optimizer.zero_grad()

                # Move training data to the device
                seq = X_train[j].unsqueeze(0).to(self.device)
                
                signal_pred, tp_pred, sl_pred = self(seq)
                
                # And the labels
                signal_target = y_train_signal[j].unsqueeze(0).to(self.device)
                tp_target = y_train_tp[j].to(self.device)
                sl_target = y_train_sl[j].to(self.device)

                # Calculate losses (per-sample)
                signal_loss = signal_loss_func(signal_pred, signal_target)
                tp_loss = price_loss_func(tp_pred, tp_target.unsqueeze(0))  # Match dimensions
                sl_loss = price_loss_func(sl_pred, sl_target.unsqueeze(0))  # Match dimensions
                
                # Apply time weight to this sample
                weight = time_weights[j]
                weighted_loss = weight * (signal_loss.mean() + tp_loss.mean() + sl_loss.mean())
                
                weighted_loss.backward()
                optimizer.step()
                
                total_loss += weighted_loss.item()

            if (i+1) % 1 == 0:
                avg_loss   = total_loss / len(X_train)
                print(f'epoch: {i+1:3} loss: {avg_loss:10.8f}')
    
    def predict(self, X_test):
        self.eval()
        with torch.no_grad():
            # 4. Move prediction data to the device
            X_test = X_test.to(self.device)
            signal_probs, tp_pred, sl_pred = self(X_test)
            return torch.argmax(signal_probs, dim=1), signal_probs, tp_pred, sl_pred

    def save_model(self, filepath='crypto_predicter_model.pth'):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath='crypto_predicter_model.pth'):
        # 5. Ensure model loads to the correct device
        self.load_state_dict(torch.load(filepath, map_location=self.device))
