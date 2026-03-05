import torch
import torch.nn as nn

class HGDM(nn.Module):
    """
    Head-Gaze Dynamics Model (HGDM)
    Expert model to learn spatio-temoral correlation between head pose and gaze.
    Learns patterns like 'pose-gaze decoupling' (e.g., phone usage).
    """
    def __init__(self, input_dim=7, hidden_dim=64, num_layers=2, dropout=0.3):
        super(HGDM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout, 
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output for classification (or mean/pooling)
        # For temporal signals, we often want the prediction at each step or final
        out = self.fc(lstm_out)
        return self.sigmoid(out)

if __name__ == "__main__":
    model = HGDM()
    test_input = torch.randn(8, 120, 7)
    output = model(test_input)
    print(f"HGDM Output shape: {output.shape}") # Expect (8, 120, 1)
