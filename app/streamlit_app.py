import argparse
from pathlib import Path

from PIL import Image
import streamlit as st

from captioning.inference.infer import load_generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=str, default="artifacts/caption_model.pt")
    parser.add_argument("--vocab", type=str, default="artifacts/vocab.json")
    known_args, _ = parser.parse_known_args()
    return known_args


ARGS = parse_args()


@st.cache_resource

def get_generator(checkpoint_path: str, vocab_path: str):
    """I lazy-load and cache the model so repeated predictions stay fast."""
    return load_generator(checkpoint_path=checkpoint_path, vocab_path=vocab_path, device="cpu")


def main() -> None:
    st.set_page_config(page_title="Image Caption Generator", page_icon="🖼️")
    st.title("🖼️ Image Caption Generator")
    st.write("Upload an image and I will generate a caption using a CNN + LSTM model.")

    checkpoint_path = st.sidebar.text_input("Checkpoint path", ARGS.checkpoint)
    vocab_path = st.sidebar.text_input("Vocab path", ARGS.vocab)
    max_length = st.sidebar.slider("Max caption length", min_value=5, max_value=40, value=20, step=1)

    if not Path(checkpoint_path).exists() or not Path(vocab_path).exists():
        st.warning("Checkpoint or vocab file not found. Train the model first.")
        return

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    generator = get_generator(checkpoint_path, vocab_path)
    caption = generator.generate(image, max_length=max_length)

    st.subheader("Generated caption")
    st.success(caption or "No caption generated")


if __name__ == "__main__":
    main()
