import torch
import numpy as np

DEVICE = torch.device("cpu")

class UC4DriftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=257,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = torch.nn.Linear(256, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        final_state = out[:, -1, :]
        logit = self.fc(final_state)
        return logit.squeeze(1)


class UC4Engine:
    def __init__(self, model_path):
        self.model = UC4DriftModel().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        self.sequence_buffer = []

    def reset(self):
        self.sequence_buffer.clear()

    def update(self, delta_vector, cosine_similarity):
        feature = np.concatenate([delta_vector, [cosine_similarity]])
        self.sequence_buffer.append(feature)

        if len(self.sequence_buffer) < 120:
            return 0.0

        if len(self.sequence_buffer) > 120:
            self.sequence_buffer.pop(0)

        sequence = torch.tensor(
            np.array(self.sequence_buffer),
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logit = self.model(sequence)
            prob = torch.sigmoid(logit).item()

        return prob