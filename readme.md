# tiny-DDPM

A minimal and extensible implementation of Diffusion Models for the MNIST dataset. This project supports core techniques like **DDPM**, **DDIM**, and **Classifier-Free Guidance**, featuring both traditional **UNet** and modern **Diffusion Transformer (DiT)** architectures.

## ✨ Features

- **Multiple Architectures**: Switch between `UNet` and `Diffusion Transformer (DiT)` via CLI arguments.
- **Config-Driven**: Manage model hyperparameters and training settings through `config.yaml`.
- **CFG Support**: Implementation of Classifier-Free Guidance for conditional generation.
- **Sampling Algorithms**: Support for both DDPM (stochastic) and DDIM (deterministic/accelerated) sampling.
- **Apple Silicon Optimized**: Automatic acceleration using MPS on M-series chips (CUDA/CPU also supported).
- **Interactive GUI**: A Streamlit-based interface to visualize the denoising process step-by-step.

## 📁 Project Structure

```
tiny-DDPM/
├── config.yaml              # Model and training configuration
├── src/
│   ├── train.py             # Main training script
│   ├── sampling.py          # Image generation script
│   ├── gui.py               # Streamlit-based visualization GUI
│   └── tiny_DDPM/           # Core library
│       ├── diffusion.py     # Main diffusion orchestrator
│       ├── forward_encoder.py # Forward process (noise addition)
│       ├── reverse_decoder.py # Reverse process (DDPM/DDIM sampling)
│       ├── noise_sheduler.py # Noise schedule (alpha/beta) management
│       ├── utils.py         # Utility functions
│       └── modules/         # Neural network architectures
│           ├── unet.py      # U-Net
│           ├── dit.py       # Diffusion Transformer
│           └── layer.py     # Shared layers and embeddings
├── scripts/                 # Automation scripts
│   ├── train.sh
│   └── inference.sh
└── assets/                  # Results and screenshots
```

## ⚙️ Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/tiny-DDPM.git
   cd tiny-DDPM
   ```

2. **Install dependencies**
   Using `uv` (recommended) or `pip`:
   ```bash
   # Using uv
   uv sync

   # Using pip
   pip install -e .
   ```

## 🚀 Training

Train your model by specifying the architecture and hyperparameters. Configurations are managed in `config.yaml`, but can be overridden via command-line arguments.

**Example Commands:**

- **Train UNet**
  ```bash
  python3 src/train.py --model-type UNet --epochs 30 --batch-size 8 --lr 0.0001
  ```

- **Train DiT**
  ```bash
  python3 src/train.py --model-type DiT --epochs 50 --batch-size 4 --lr 0.0002
  ```

### Automation Script
Use the provided shell script to run pre-defined training routines:
```bash
chmod +x scripts/train.sh
./scripts/train.sh
```

## 🎨 Sampling

Generate images from a trained checkpoint. You must provide the model type and the path to the weight file (`.pt`).

**Example Command:**
```bash
python3 src/sampling.py --model-type UNet --model-path "UNet_T1000_E30.pt"
```

### Automation Script
```bash
chmod +x scripts/inference.sh
./scripts/inference.sh
```

## 🎮 Interactive GUI

Visualize the denoising process in real-time using Streamlit.

```bash
streamlit run src/gui.py
```

The GUI allows you to:
- Select the digit to generate (0-9).
- Choose between UNet and DiT.
- Load specific checkpoints.
- Adjust guidance weight (CFG) and sampling steps.
- **Scrub through the denoising steps** to see how the image emerges from noise.

![GUI Screenshot](./assets/screenshot.png)

## 📊 Results

Generated using DDIM (`steps=10`) with Classifier-Free Guidance (`w=1.0`).

<p align="center">
  <img src="./assets/result.png" width="30%" />
  <img src="./assets/result2.png" width="30%" />
  <img src="./assets/result3.png" width="30%" />
</p>

### Training Loss
![loss](./assets/loss.png)

## 📚 References

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
