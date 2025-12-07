# ComfyUI-MAP-Probe (v0.1)

**A Geometric "Vector Network Analyzer" for Stable Diffusion Latent Trajectories.**

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

This custom node implements the **Manifold Alignment Protocol (MAP)** within ComfyUI. It transforms the "black box" of diffusion sampling into a measurable, visualizable geometric process.

Instead of guessing whether `Step=40` is better than `Step=30`, MAP-Probe quantifies the **Semantic Confidence (Depth)** and **Structural Stability (Convergence)** of the generation process, providing scientific feedback for prompt engineering and hyperparameter tuning.

For the theoretical foundation, please visit the main repository:
👉 [The Manifold Alignment Protocol (MAP)](https://github.com/JBKing514/map_blog)

---

## ⚠️ Important: Installation Requirement

**This node requires `matplotlib` to generate visualization plots.**

You **MUST** install it in the Python environment used by ComfyUI.

### For Windows Portable Users:
Open a terminal in your `ComfyUI_windows_portable` folder and run:
```cmd
.\python_embeded\python.exe -m pip install matplotlib
```

### For Standard venv/Conda Users:
Activate your environment and run:
```bash
pip install matplotlib
```

---

## Features

### 🔬 Manual Analysis (The VNA Mode)
Acts like an oscilloscope for your diffusion process.
- **Differential Tracking:** Automatically detects if you are using the same seed as the previous run.
- **Visual Feedback:** Displays **Green (Improvement)** or **Red (Regression)** to show exactly how your prompt or parameter tweaks affect generation quality.
- **Quantified Metrics:**
    - **Depth:** How deep the latent vector penetrated the semantic manifold (Signal Strength).
    - **Stability:** Whether the trajectory stabilized at the attractor well (Convergence Quality).

### 🤖 Auto-Tuner (Hill Climbing)
A "Self-Driving" mode that optimizes parameters to save compute.
- **Step Search:** Automatically iterates steps (e.g., 20, 25, 30...) to find the peak Q-score.
- **Early Stopping:** Detects "Over-baking" (when quality starts to drop despite more steps) and stops automatically.
- **CFG Refinement:** (Optional) Fine-tunes CFG scale around the optimal step count.

### 📊 Data Logging
- **Auto-CSV:** Successfully optimized runs are automatically logged to `ComfyUI/output/MAP_Tuning_Log.csv` with timestamps, parameters, and Q-scores.
- **Prompt Archiving:** Connect your prompt text to `optional_positive_text` to build a "Golden Prompt" database.

---

## Usage

### 1. Basic Setup
Search for the node **"MAP Pro Suite"**. It functions as a replacement for the standard `KSampler`.

- **Input:** Connect Model, Positive/Negative Conditioning, and Empty Latent.
- **Output:**
    - `LATENT`: The best latent found (passed to VAE Decode).
    - `IMAGE`: The trajectory analysis plot (connect to `Preview Image`).
    - `STRING`: Detailed analysis text (connect to `Show Text` from *ComfyUI-Custom-Scripts*).

### 2. Operation Modes

#### Mode A: `Analyze (Manual)`
Best for exploration and manual tweaking.
1.  Run with a **Random Seed**. The node establishes a reference Q-score.
2.  Switch to **Fixed Seed**.
3.  Tweak `Steps` or `CFG` or modify your Prompt.
4.  Run again. The plot will show a **Δ (Delta)** indicating improvement or degradation.

#### Mode B: `Auto-Tune (Hill Climb)`
Best for production and optimization.
1.  Set `tuner_max_steps` (e.g., 50) and `tuner_stride` (e.g., 5).
2.  The node will run a loop internally (e.g., 20 -> 25 -> 30...).
3.  It returns the **Best Latent** found before quality peaked or dropped.
4.  Check `ComfyUI/output/MAP_Tuning_Log.csv` for the record.

---

## Visual Examples

### Manual Analysis: Improvement vs Regression

<p align="center">
  <img src="examples/Manual_Imp.png" width="500" alt="MAP Analysis Improved">
</p>

<p align="center">
  <img src="examples/Manual_Reg.png" width="500" alt="MAP Analysis Regress">
</p>

### Auto-Tuning Curve

<p align="center">
  <img src="examples/Auto_Tuner.png" width="500" alt="MAP Auto Tune">
</p>

---

## Citation

If you use this toolkit or MAP in your research, please cite:

```bibtex
@article{tang2025map,
  title={The Manifold Alignment Protocol (MAP): A Self-Iterable Geometric Framework for Cross-System Cognitive Convergence},
  author={Tang, Yunchong},
  journal={arXiv preprint arXiv:2511.xxxxx},
  year={2025}
}
```

## License

MIT License © Yunchong Tang