# Contributing to ComfyUI-MAP-Probe

Thank you for your interest in contributing!
This project brings the geometric insights of the Manifold Alignment Protocol (MAP) to the practical world of ComfyUI. We welcome improvements to the algorithms, UI enhancements, and compatibility fixes.

---

## How to Contribute

### 1. Fork the Repository
Create your own fork and clone it into your `ComfyUI/custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/JBKing514/map_comfyui.git
```

### 2. Development Workflow

Since this is a ComfyUI node, testing requires running ComfyUI.
- **Hot Reloading:** We recommend creating a workflow that uses the node, and restarting ComfyUI (or reloading via the Manager) to test changes.
- **Dependencies:** Ensure your environment has `matplotlib` installed.

### 3. Add Your Changes

The core logic resides in `map_nodes.py`.
- **Algorithm Logic:** Modifications to Q-score calculation or projection should be well-commented.
- **UI/Plotting:** Changes to `matplotlib` plotting code should ensure the text is legible and colors are distinct.

---

## Coding Style Guidelines

- **Python:** Follow PEP8 where possible.
- **Type Hints:** Use type hints for function arguments in the core logic.
- **Error Handling:** Since KSampler interactions can be fragile, ensure proper `try/except` blocks to prevent crashing the entire ComfyUI process.

---

## Submitting a Pull Request (PR)

### Before submitting:

1.  **Test Manual Mode:** Ensure differential tracking works when switching seeds.
2.  **Test Auto-Tuner:** Ensure it stops correctly at the peak and doesn't run infinitely.
3.  **Check Logging:** Verify that CSVs are written correctly to the output folder.

### Then open a PR with:

-   A clear title.
-   A short description of the change.
-   **Screenshots** of the node output (Graphs/Plots) showing the effect of your changes.
-   
**note:** This project is currently maintained by one person (a full-time PhD student),  
so PR reviews may take time. Thanks for your patience!

---

## Reporting Issues

If you find a bug (e.g., tensor shape mismatches with certain SDXL/SD1.5 models):

1.  Open a GitHub Issue.
2.  Include your **ComfyUI Console Log** (Traceback).
3.  State which Model/Resolution you were using.

## Roadmap

We are looking for help with:

- Support for other Samplers (Ancestral, SDE).
- 3D Trajectory Visualization (Interactive HTML).
- Integration with ComfyUI-Manager for auto-installing dependencies.

Thank You!

— Yunchong Tang