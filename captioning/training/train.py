import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from captioning.models.decoder import DecoderLSTM
from captioning.models.encoder import EncoderCNN
from captioning.training.dataset import Flickr8kDataset, build_vocab_from_file, collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train image captioning model on Flickr8k")
    parser.add_argument("--images-dir", type=str, default="data/raw/Flickr8k_Dataset")
    parser.add_argument("--captions-file", type=str, default="data/raw/Flickr8k.token.txt")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args()


def train_one_epoch(
    encoder: EncoderCNN,
    decoder: DecoderLSTM,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_idx: int,
) -> float:
    """I run one training pass with teacher forcing and ignore padded targets in loss."""
    encoder.train()
    decoder.train()
    running_loss = 0.0

    for images, captions, _ in dataloader:
        images = images.to(device)
        captions = captions.to(device)

        targets = captions
        image_features = encoder(images)
        logits = decoder(image_features, captions)

        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)

        valid_positions = targets != pad_idx
        masked_logits = logits[valid_positions]
        masked_targets = targets[valid_positions]

        loss = criterion(masked_logits, masked_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

    return running_loss / max(1, len(dataloader))


def main() -> None:
    args = parse_args()
    os.makedirs(args.artifacts_dir, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    vocab = build_vocab_from_file(args.captions_file, min_freq=args.min_freq)
    vocab_path = os.path.join(args.artifacts_dir, "vocab.json")
    vocab.save(vocab_path)

    dataset = Flickr8kDataset(
        images_dir=args.images_dir,
        captions_file=args.captions_file,
        vocab=vocab,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn(vocab.pad_idx),
    )

    encoder = EncoderCNN(embed_size=args.embed_size).to(device)
    decoder = DecoderLSTM(
        vocab_size=len(vocab.stoi),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)

    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            pad_idx=vocab.pad_idx,
        )
        print(f"Epoch [{epoch + 1}/{args.epochs}] - loss: {avg_loss:.4f}")

    checkpoint = {
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "embed_size": args.embed_size,
        "hidden_size": args.hidden_size,
        "vocab_size": len(vocab.stoi),
    }
    checkpoint_path = os.path.join(args.artifacts_dir, "caption_model.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved vocab: {vocab_path}")


if __name__ == "__main__":
    main()
