# File: CONTRIBUTING.md

# Contributing to MAP-ComfyUI

Thank you for your interest in contributing!
This project brings the geometric insights of the Manifold Alignment Protocol (MAP) to the practical world of ComfyUI. We welcome improvements to the algorithms, UI enhancements, and compatibility fixes.

---

## How to Contribute

### 1. Fork the Repository
Create your own fork and clone it into your `ComfyUI/custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes/
git clone [https://github.com/JBKing514/map_comfyui.git](https://github.com/JBKing514/map_comfyui.git)
```

### 2. Development Workflow

Since this is a ComfyUI node, testing requires running ComfyUI.
- **Hot Reloading:** We recommend creating a workflow that uses the node, and restarting ComfyUI (or reloading via the Manager) to test changes.
- **Dependencies:** Ensure your environment has `matplotlib` installed.
- **Plotting Logic:** We use `matplotlib.gridspec` for the dashboard layout. When modifying `plot_tuning_curve`, please ensure the bottom 1/6th of the figure remains reserved for the text dashboard to prevent overlap.

### 3. Add Your Changes

The core logic resides in `map_nodes.py`.
- **Algorithm Logic:** Modifications to Q-score calculation or projection should be well-commented.
- **Scheduler Optimization:** Logic for scheduler sweeping is in `run_auto_tuner` (Phase 2). Be careful with extensive loops as they consume GPU time.

---

## Coding Style Guidelines

- **Python:** Follow PEP8 where possible.
- **Type Hints:** Use type hints for function arguments in the core logic.
- **Error Handling:** Since KSampler interactions can be fragile, ensure proper `try/except` blocks to prevent crashing the entire ComfyUI process.

---

## Submitting a Pull Request (PR)

### Before submitting:

1.  **Test Manual Mode:** Ensure differential tracking works when switching seeds.
2.  **Test Auto-Tuner:** - Verify Phase 1 (Steps), Phase 2 (Scheduler), and Phase 3 (CFG) execute in order.
    - Ensure the Dashboard text (Bottom panel) aligns correctly and doesn't clip.
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

- **Smart Caching:** Implementing a hash-based cache to skip Scheduler Optimization if the Model+Steps combination has been seen before (Performance).
- **3D Trajectory Visualization:** Interactive HTML export of the manifold trajectory.
- **Profiling Mode:** A "One-Click" analysis mode that runs a fixed battery of tests on a new Checkpoint to generate a recommended settings report.
- **Integration:** ComfyUI-Manager auto-install support.

Thank You!

— Yunchong Tang