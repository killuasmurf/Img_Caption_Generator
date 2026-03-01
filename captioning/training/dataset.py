import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms


SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]


def normalize_text(text: str) -> List[str]:
    """I lowercase text and keep only alpha-numeric tokens for a tiny clean vocabulary."""
    text = text.lower().strip()
    tokens = re.findall(r"[a-z0-9']+", text)
    return tokens


@dataclass
class Vocabulary:
    stoi: Dict[str, int]
    itos: Dict[int, str]
    pad_idx: int
    start_idx: int
    end_idx: int
    unk_idx: int

    @classmethod
    def build(cls, captions: List[str], min_freq: int = 2) -> "Vocabulary":
        """I build a small vocabulary from caption text and keep infrequent words as <unk>."""
        counter = Counter()
        for caption in captions:
            counter.update(normalize_text(caption))

        words = [w for w, freq in counter.items() if freq >= min_freq]
        ordered_words = SPECIAL_TOKENS + sorted(words)
        stoi = {word: idx for idx, word in enumerate(ordered_words)}
        itos = {idx: word for word, idx in stoi.items()}
        return cls(
            stoi=stoi,
            itos=itos,
            pad_idx=stoi["<pad>"],
            start_idx=stoi["<start>"],
            end_idx=stoi["<end>"],
            unk_idx=stoi["<unk>"],
        )

    def encode(self, caption: str) -> List[int]:
        """I convert a caption to token IDs and wrap it with start/end tokens."""
        tokens = normalize_text(caption)
        token_ids = [self.stoi.get(token, self.unk_idx) for token in tokens]
        return [self.start_idx] + token_ids + [self.end_idx]

    def decode(self, token_ids: List[int]) -> str:
        """I decode token IDs into text and stop once I hit <end>."""
        words = []
        for idx in token_ids:
            word = self.itos.get(int(idx), "<unk>")
            if word == "<end>":
                break
            if word in {"<start>", "<pad>"}:
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path: str) -> None:
        """I save vocab indices to JSON so inference can load the exact same mapping."""
        payload = {
            "stoi": self.stoi,
            "pad_idx": self.pad_idx,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "unk_idx": self.unk_idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """I restore a saved vocabulary mapping from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        stoi = payload["stoi"]
        itos = {idx: token for token, idx in stoi.items()}
        return cls(
            stoi=stoi,
            itos=itos,
            pad_idx=payload["pad_idx"],
            start_idx=payload["start_idx"],
            end_idx=payload["end_idx"],
            unk_idx=payload["unk_idx"],
        )


def load_flickr_captions(captions_file: str) -> List[Tuple[str, str]]:
    """I parse Flickr8k caption files and return (image_name, caption) pairs."""
    pairs = []
    with open(captions_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if "\t" in line:
                image_part, caption = line.split("\t", 1)
            else:
                parts = line.split(",", 1)
                if len(parts) < 2:
                    continue
                image_part, caption = parts[0], parts[1]

            if "#" in image_part:
                image_name = image_part.split("#", 1)[0]
            else:
                image_name = image_part

            pairs.append((image_name, caption.strip()))
    return pairs


class Flickr8kDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        vocab: Vocabulary,
        transform: transforms.Compose | None = None,
        max_samples: int | None = None,
    ) -> None:
        """I map each image-caption pair to tensors for encoder/decoder training."""
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        pairs = load_flickr_captions(captions_file)
        self.samples = pairs[:max_samples] if max_samples else pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_name, caption = self.samples[idx]
        image_path = os.path.join(self.images_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        caption_ids = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return image_tensor, caption_ids


def build_vocab_from_file(captions_file: str, min_freq: int = 2) -> Vocabulary:
    """I build vocabulary directly from a Flickr8k caption file."""
    pairs = load_flickr_captions(captions_file)
    captions = [caption for _, caption in pairs]
    return Vocabulary.build(captions=captions, min_freq=min_freq)


def collate_fn(pad_idx: int):
    """I pad variable-length captions so each batch becomes a single tensor."""

    def _collate(batch):
        images, captions = zip(*batch)
        image_tensor = torch.stack(images, dim=0)
        lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
        caption_tensor = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
        return image_tensor, caption_tensor, lengths

    return _collate
