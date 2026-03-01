import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """I decode image features into word logits using an embedding layer + LSTM."""
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """I apply teacher forcing by feeding image feature token then ground-truth caption tokens."""
        caption_embeddings = self.embedding(captions[:, :-1])
        image_step = image_features.unsqueeze(1)
        lstm_input = torch.cat((image_step, caption_embeddings), dim=1)
        outputs, _ = self.lstm(lstm_input)
        logits = self.fc(outputs)
        return logits

    def greedy_decode(
        self,
        image_features: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_length: int = 20,
    ) -> list[int]:
        """I generate tokens one-by-one using greedy argmax decoding."""
        generated = [start_idx]
        states = None

        inputs = image_features.unsqueeze(1)
        for _ in range(max_length):
            outputs, states = self.lstm(inputs, states)
            logits = self.fc(outputs.squeeze(1))
            next_token = int(torch.argmax(logits, dim=-1).item())
            generated.append(next_token)
            if next_token == end_idx:
                break
            inputs = self.embedding(torch.tensor([[next_token]], device=image_features.device))

        return generated
