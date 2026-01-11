import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer to inject temporal information.
    Uses sinusoidal encoding as in Vaswani et al. 2017.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model) or (batch, seq_len, d_model)
        """
        if x.dim() == 3 and x.size(0) == self.pe.size(0):
            # (seq_len, batch, d_model) format
            x = x + self.pe[:x.size(0), :]
        elif x.dim() == 3:
            # (batch, seq_len, d_model) format - need to transpose
            seq_len = x.size(1)
            x = x + self.pe[:seq_len, :].transpose(0, 1)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-head self-attention and feed-forward network.
    This replaces the LSTM's sequential processing with parallel attention mechanism.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        # Multi-head self-attention with residual connection
        src2 = self.norm1(src)
        attn_output, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask)
        src = src + self.dropout1(attn_output)
        
        # Feed-forward with residual connection
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class CryptoPredicter(nn.Module):
    """
    Transformer-based crypto price predictor.
    
    Advantages over LSTM:
    1. No forgetting phenomenon - all timesteps attended to simultaneously
    2. Parallel computation capability
    3. All-to-all attention mechanism captures long-range dependencies
    4. Better handling of temporal patterns through positional encoding
    """
    def __init__(self, input_size=93, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.25, max_seq_len=5000,
                 hidden_layer_size=None, use_dropout_inference=True, **kwargs):
        """
        Transformer-based crypto predictor initialization.
        
        Args:
            input_size: Number of input features (technical indicators)
            d_model: Model dimension (embedding size)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feed-forward network dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            hidden_layer_size: (deprecated) For backward compatibility, ignored
            use_dropout_inference: If True, applies dropout during inference for uncertainty estimation
            **kwargs: Additional arguments for backward compatibility
        """ 
        super().__init__()
        # 1. Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Transformer model configured to use device: {self.device}")
        print(f"🧠 Model input size: {input_size} features (technical indicators)")
        print(f"🧠 Transformer config: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        
        self.d_model = d_model
        self.input_size = input_size
        self.use_dropout_inference = use_dropout_inference
        
        # Input projection: map features to model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding to inject temporal information
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer encoder layers - stack multiple encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Global pooling: use attention-weighted pooling to aggregate sequence
        self.pooling_attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Output heads with temperature scaling for better calibration
        self.signal_head = nn.Linear(d_model, 2)  # 0: sell, 1: buy
        self.tp_head = nn.Linear(d_model, 1)
        self.sl_head = nn.Linear(d_model, 1)
        
        # Temperature parameter for confidence calibration (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Additional dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # Store dropout rate for verification
        self.dropout_rate = dropout
        
        # Time decay factor for pooling attention (bias toward recent timesteps)
        self.time_decay_alpha = 3.0
        
        # 2. Move model to the selected device
        self.to(self.device)

    def forward(self, input_seq):
        """
        Forward pass through transformer encoder.
        
        Args:
            input_seq: Tensor of shape (batch, seq_len, input_size)
        
        Returns:
            signal_probs: (batch, 2) - BUY/SELL probabilities
            take_profit: (batch, 1) - Predicted take profit price
            stop_loss: (batch, 1) - Predicted stop loss price
        """
        batch_size, seq_len, input_size = input_seq.shape
        
        # Project input features to model dimension
        x = self.input_projection(input_seq)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)
        
        # Apply dropout
        x = self.dropout_layer(x)
        
        # Pass through transformer encoder layers (parallel processing of all timesteps)
        # Transformer processes entire sequence simultaneously, no sequential dependency
        transformer_out = x
        for layer in self.transformer_layers:
            transformer_out = layer(transformer_out)  # (batch, seq_len, d_model)
        
        # Global pooling: attention-weighted aggregation of sequence
        # This allows the model to focus on the most relevant timesteps
        attention_scores = self.pooling_attention(transformer_out)  # (batch, seq_len, 1)
        
        # Add time-decay bias to favor more recent timesteps
        positions = torch.linspace(0, 1, seq_len, device=self.device)  # 0 (oldest) to 1 (most recent)
        time_bias = torch.exp(self.time_decay_alpha * (positions - 1))  # Exponential decay
        time_bias = time_bias.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # Combine learned attention with time decay bias
        attention_scores = attention_scores + time_bias
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Weighted sum to get final representation
        pooled_output = torch.sum(transformer_out * attention_weights, dim=1)  # (batch, d_model)
        
        # Apply dropout before output heads for MC dropout variation
        pooled_output = self.dropout_layer(pooled_output)
        
        # Generate predictions from pooled representation
        signal_logits = self.signal_head(pooled_output)
        take_profit = self.tp_head(pooled_output)
        stop_loss = self.sl_head(pooled_output)
        
        # Apply temperature scaling for better calibration
        # Higher temperature -> softer probabilities, lower confidence
        scaled_logits = signal_logits / self.temperature
        
        return F.softmax(scaled_logits, dim=1), take_profit, stop_loss

    def train_model(self, X_train, y_train_signal, y_train_tp, y_train_sl, epochs=10, lr=0.001, time_weighted=True, batch_size=32, label_smoothing=0.1):
        """
        Train the model with optional time-weighted loss and batch training.
        
        Args:
            time_weighted: If True, recent samples get higher weight (exponential decay from most recent)
            batch_size: Number of samples per batch for more stable training
            label_smoothing: Label smoothing factor to prevent overconfidence (0.0 to 0.5)
        """
        # Use label smoothing to prevent overconfidence
        signal_loss_func = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
        price_loss_func  = nn.MSELoss(reduction='none')
        optimizer        = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)  # Add L2 regularization
        
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

        print(f"🎯 Training with batch_size={batch_size}, label_smoothing={label_smoothing}")
        
        for i in range(epochs):
            self.train()  # Ensure model is in training mode
            total_loss = 0
            num_batches = 0
            
            # Shuffle indices for each epoch
            indices = torch.randperm(len(X_train))
            
            # Process in batches
            for start_idx in range(0, len(X_train), batch_size):
                end_idx = min(start_idx + batch_size, len(X_train))
                batch_indices = indices[start_idx:end_idx]
                
                optimizer.zero_grad()

                # Move batch to device
                seq_batch = X_train[batch_indices].to(self.device)
                signal_target_batch = y_train_signal[batch_indices].to(self.device)
                tp_target_batch = y_train_tp[batch_indices].to(self.device)
                sl_target_batch = y_train_sl[batch_indices].to(self.device)
                
                # Forward pass
                signal_pred, tp_pred, sl_pred = self(seq_batch)
                
                # Calculate losses
                signal_loss = signal_loss_func(signal_pred, signal_target_batch)
                tp_loss = price_loss_func(tp_pred.squeeze(), tp_target_batch.squeeze())
                sl_loss = price_loss_func(sl_pred.squeeze(), sl_target_batch.squeeze())
                
                # Apply time weights to batch
                batch_weights = time_weights[batch_indices]
                weighted_signal_loss = (signal_loss * batch_weights).mean()
                weighted_tp_loss = (tp_loss * batch_weights).mean()
                weighted_sl_loss = (sl_loss * batch_weights).mean()
                
                # Total weighted loss
                weighted_loss = weighted_signal_loss + weighted_tp_loss + weighted_sl_loss
                
                weighted_loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += weighted_loss.item()
                num_batches += 1

            if (i+1) % 1 == 0:
                avg_loss = total_loss / num_batches
                print(f'epoch: {i+1:3} loss: {avg_loss:10.8f}')
    
    def predict(self, X_test, mc_samples=10):
        """
        Predict with Monte Carlo dropout for uncertainty estimation.
        
        Args:
            X_test: Input tensor
            mc_samples: Number of forward passes with dropout enabled for uncertainty estimation
        
        Returns:
            predictions: Argmax of averaged probabilities
            signal_probs: Mean probabilities across MC samples
            tp_pred: Mean TP prediction
            sl_pred: Mean SL prediction
            uncertainty: Standard deviation of probabilities (uncertainty measure)
        """
        # Move prediction data to the device
        X_test = X_test.to(self.device)
        
        if self.use_dropout_inference and mc_samples > 1:
            # Monte Carlo Dropout: Keep dropout enabled during inference
            # to get uncertainty estimates
            self.train()  # Enable dropout - CRITICAL: this enables dropout layers
            
            # Verify dropout is enabled
            if not self.training:
                import warnings
                warnings.warn("Model is not in training mode! MC dropout may not work correctly.")
            
            all_signal_probs = []
            all_tp_preds = []
            all_sl_preds = []
            
            with torch.no_grad():
                for i in range(mc_samples):
                    signal_probs, tp_pred, sl_pred = self(X_test)
                    all_signal_probs.append(signal_probs)
                    all_tp_preds.append(tp_pred)
                    all_sl_preds.append(sl_pred)
            
            # Average predictions across MC samples
            signal_probs_stacked = torch.stack(all_signal_probs)
            mean_signal_probs = signal_probs_stacked.mean(dim=0)
            uncertainty = signal_probs_stacked.std(dim=0)  # Uncertainty estimate
            
            # Diagnostic: Check if MC dropout is creating variation
            max_probs_per_sample = signal_probs_stacked.max(dim=-1)[0]  # Max prob for each sample
            prob_variance = max_probs_per_sample.var().item()
            
            # Log warning if variance is too low (MC dropout not working)
            if prob_variance < 1e-6:
                import warnings
                warnings.warn(f"MC Dropout Warning: Very low variance ({prob_variance:.2e}) across {mc_samples} samples. "
                            f"Dropout may not be active. Check model.training={self.training}, "
                            f"use_dropout_inference={self.use_dropout_inference}")
            
            mean_tp_pred = torch.stack(all_tp_preds).mean(dim=0)
            mean_sl_pred = torch.stack(all_sl_preds).mean(dim=0)
            
            predictions = torch.argmax(mean_signal_probs, dim=1)
            
            # Reset to eval mode after MC sampling
            self.eval()
            
            return predictions, mean_signal_probs, mean_tp_pred, mean_sl_pred, uncertainty
        else:
            # Standard prediction without dropout
            self.eval()
            with torch.no_grad():
                signal_probs, tp_pred, sl_pred = self(X_test)
                predictions = torch.argmax(signal_probs, dim=1)
                # Return zero uncertainty when not using MC dropout
                uncertainty = torch.zeros_like(signal_probs)
                return predictions, signal_probs, tp_pred, sl_pred, uncertainty

    def save_model(self, filepath='crypto_predicter_model.pth'):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath='crypto_predicter_model.pth'):
        # 5. Ensure model loads to the correct device
        self.load_state_dict(torch.load(filepath, map_location=self.device))
