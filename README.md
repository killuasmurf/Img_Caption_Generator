# Image Caption Generator (PyTorch + Streamlit)

Minimal, reproducible image caption generator using a pretrained ResNet18 encoder and LSTM decoder trained on Flickr8k.

## Project Structure

```text
.
├── app/
│   └── streamlit_app.py
├── artifacts/
│   └── .gitkeep
├── captioning/
│   ├── inference/
│   │   └── infer.py
│   ├── models/
│   │   ├── decoder.py
│   │   └── encoder.py
│   └── training/
│       ├── dataset.py
│       └── train.py
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Setup (Flickr8k)

Place Flickr8k files in:

```text
data/raw/
├── Flickr8k_Dataset/
│   ├── 1000268201_693b08cb0e.jpg
│   └── ...
└── Flickr8k.token.txt
```

## Training (CPU default)

Run a quick 1-epoch training pass:

```bash
python -m captioning.training.train \
  --images-dir data/raw/Flickr8k_Dataset \
  --captions-file data/raw/Flickr8k.token.txt \
  --artifacts-dir artifacts \
  --epochs 1 \
  --batch-size 16 \
  --cpu
```

Optional quick smoke run with a subset:

```bash
python -m captioning.training.train \
  --images-dir data/raw/Flickr8k_Dataset \
  --captions-file data/raw/Flickr8k.token.txt \
  --artifacts-dir artifacts \
  --epochs 1 \
  --batch-size 8 \
  --max-samples 64 \
  --cpu
```

Artifacts generated:

- `artifacts/caption_model.pt`
- `artifacts/vocab.json`

## Run Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Optional custom artifact paths:

```bash
streamlit run app/streamlit_app.py -- \
  --checkpoint artifacts/caption_model.pt \
  --vocab artifacts/vocab.json
```

## Inference Behavior

- Upload image in Streamlit UI
- Model loads lazily and is cached
- Caption generated via greedy decoding
- Max generation length configurable from sidebar


## Troubleshooting Dependency Installation

If `pip install -r requirements.txt` fails with proxy or network errors (for example `Tunnel connection failed: 403 Forbidden`), your environment blocks outbound package downloads.

In that case, use one of these options:

```bash
# Option 1: configure your proxy for pip
export HTTPS_PROXY=http://<proxy-host>:<proxy-port>
export HTTP_PROXY=http://<proxy-host>:<proxy-port>
pip install -r requirements.txt
```

```bash
# Option 2: install from pre-downloaded wheel files
pip install --no-index --find-links /path/to/wheels -r requirements.txt
```

```bash
# Option 3: if your platform needs a specific PyTorch wheel index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install streamlit pillow
```
