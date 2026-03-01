from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image
import torch
from torchvision import transforms

from captioning.models.decoder import DecoderLSTM
from captioning.models.encoder import EncoderCNN
from captioning.training.dataset import Vocabulary


@dataclass
class CaptionGenerator:
    encoder: EncoderCNN
    decoder: DecoderLSTM
    vocab: Vocabulary
    device: torch.device

    def __post_init__(self) -> None:
        """I keep transforms next to the model so image preprocessing stays consistent."""
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.encoder.eval()
        self.decoder.eval()

    @torch.no_grad()
    def generate(self, image: Image.Image, max_length: int = 20) -> str:
        """I convert one PIL image into a caption with greedy decoding."""
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        image_features = self.encoder(image_tensor)
        token_ids = self.decoder.greedy_decode(
            image_features=image_features,
            start_idx=self.vocab.start_idx,
            end_idx=self.vocab.end_idx,
            max_length=max_length,
        )
        return self.vocab.decode(token_ids)


def load_generator(
    checkpoint_path: str,
    vocab_path: str,
    device: Optional[str] = None,
) -> CaptionGenerator:
    """I load model weights + vocabulary into a ready-to-use generator object."""
    runtime_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = torch.load(checkpoint_path, map_location=runtime_device)
    vocab = Vocabulary.load(vocab_path)

    encoder = EncoderCNN(embed_size=checkpoint["embed_size"]).to(runtime_device)
    decoder = DecoderLSTM(
        vocab_size=checkpoint["vocab_size"],
        embed_size=checkpoint["embed_size"],
        hidden_size=checkpoint["hidden_size"],
    ).to(runtime_device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    return CaptionGenerator(encoder=encoder, decoder=decoder, vocab=vocab, device=runtime_device)
