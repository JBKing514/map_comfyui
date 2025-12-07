import torch
import numpy as np
import io
import matplotlib.pyplot as plt
import comfy.sample
import comfy.model_management
import nodes

# === 全局变量 ===
GLOBAL_PROJ_MATRIX = None
GLOBAL_PROJ_DIM = 0

class MAP_Trajectory_Sampler:
    # 历史记录缓存: {seed: q_score}
    # 只要不关闭 ComfyUI，这个记录就会一直保留，方便你反复调优
    HISTORY_CACHE = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "trajectory_plot", "analysis_text")
    FUNCTION = "sample_and_map"
    CATEGORY = "MAP_Protocol"

    def sample_and_map(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        trajectory_data = []

        # --- 1. Probe ---
        def map_callback(step, x0, x, total_steps):
            try:
                current_state = x[0].detach().cpu().flatten()
                trajectory_data.append(current_state)
            except Exception: pass

        # --- 2. Patch ---
        original_sample = comfy.sample.sample
        def my_sample_wrapper(*args, **kwargs):
            kwargs['callback'] = map_callback
            return original_sample(*args, **kwargs)
        
        comfy.sample.sample = my_sample_wrapper
        
        try:
            print(f"[MAP Protocol] Sampling (Seed={seed})...")
            samples = nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler, 
                positive, negative, latent_image, denoise=denoise
            )
        except Exception as e:
            return (latent_image, self.create_empty_image(), f"Error: {e}")
        finally:
            comfy.sample.sample = original_sample

        # --- 3. Analysis ---
        if len(trajectory_data) < 5:
            return (samples[0], self.create_empty_image(), "Insufficient steps")

        try:
            X = torch.stack(trajectory_data).float()
            T, D = X.shape

            # Global Projection (Fixed Ruler)
            global GLOBAL_PROJ_MATRIX, GLOBAL_PROJ_DIM
            if GLOBAL_PROJ_MATRIX is None or GLOBAL_PROJ_DIM != D:
                print(f"[MAP Protocol] Init Global Matrix (dim={D})...")
                P = torch.randn(D, 2)
                P = P / torch.norm(P, dim=0, keepdim=True)
                GLOBAL_PROJ_MATRIX = P
                GLOBAL_PROJ_DIM = D
            
            X_centered = X - X[0]
            coords = torch.mm(X_centered, GLOBAL_PROJ_MATRIX).numpy()

            # === Metrics (Reverted to Strict/Raw Logic) ===
            
            # 1. Depth (Force)
            dist_from_start = np.linalg.norm(coords, axis=1)
            C_raw = np.max(dist_from_start)
            # 使用较严的 5.0，保持数值在 0.2~0.7 区间，增加对比度
            lambda_val = 5.0 
            C_norm = C_raw / (C_raw + lambda_val)

            # 2. Stability (Convergence)
            velocities = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
            tail_velocity = np.mean(velocities[-5:])
            avg_velocity = np.mean(velocities) + 1e-8
            ratio = tail_velocity / avg_velocity
            # 使用较严的 5.0，严厉惩罚拖泥带水的尾部
            S_good = np.exp(-5.0 * ratio)

            # 3. Quality (Raw)
            Q_score = C_norm * S_good

            # === Differential Logic (The VNA Comparator) ===
            delta_q = 0.0
            is_first_run = True
            
            if seed in self.HISTORY_CACHE:
                last_q = self.HISTORY_CACHE[seed]
                delta_q = Q_score - last_q
                is_first_run = False
                print(f"[MAP] Seed {seed}: Old={last_q:.2%}, New={Q_score:.2%}, Delta={delta_q:+.2%}")
            else:
                print(f"[MAP] Seed {seed}: First run, cached reference.")
            
            # Update Cache
            self.HISTORY_CACHE[seed] = Q_score

            # Text Output
            if is_first_run:
                status_str = " (Ref)"
            else:
                status_str = f" ({delta_q:+.2%})"
                
            analysis_text = (
                f"MAP Differential Report (Seed {seed}):\n"
                f"====================\n"
                f"Quality: {Q_score:.2%}{status_str}\n"
                f"====================\n"
                f"Depth (Raw): {C_raw:.2f} -> {C_norm:.1%}\n"
                f"Ratio (Tail/Avg): {ratio:.2%} -> {S_good:.1%}"
            )

            plot_image = self.plot_metrics(coords, C_norm, S_good, Q_score, delta_q, is_first_run)

        except Exception as e:
            print(f"[MAP Protocol] Analysis Error: {e}")
            import traceback
            traceback.print_exc()
            plot_image = self.create_empty_image()
            analysis_text = str(e)

        return (samples[0], plot_image, analysis_text)

    def plot_metrics(self, coords, c, s, q, delta_q, is_first_run):
        plt.figure(figsize=(10, 12), dpi=100)
        gs = plt.GridSpec(5, 1)
        ax_plot = plt.subplot(gs[0:4, 0])
        ax_text = plt.subplot(gs[4, 0])
        
        x, y = coords[:, 0], coords[:, 1]
        steps = np.arange(len(x))
        
        # Plot Trajectory
        ax_plot.plot(x, y, color='gray', alpha=0.5, linestyle='--', linewidth=1)
        ax_plot.scatter(x, y, c=steps, cmap='coolwarm', s=100, edgecolors='black', zorder=10)
        ax_plot.text(x[0], y[0], ' Start', fontsize=12, va='bottom')
        ax_plot.text(x[-1], y[-1], ' End', fontsize=12, va='top', fontweight='bold')
        ax_plot.axis('equal')
        ax_plot.grid(True, alpha=0.3)
        
        # Title
        ax_plot.set_title(f"MAP Analysis (Q={q:.1%})", fontsize=14, fontweight='bold', color='#333')

        # === User Guidance Overlay ===
        if is_first_run:
            # First Run Prompt: Yellow Warning
            box_text = "First run for this seed.\nQ-score is reference only.\nFix seed to enable differential tuning."
            box_color = "#f39c12" # Yellow
            text_color = "white"
            
            ax_plot.text(0.5, 0.05, box_text, transform=ax_plot.transAxes, 
                         fontsize=12, fontweight='bold', color='black', ha='center', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.5", fc="#f1c40f", ec="none", alpha=0.8))
        else:
            # Differential Display: Comparison
            if abs(delta_q) < 0.0001:
                diff_text = "No Change"
                diff_col = "gray"
            elif delta_q > 0:
                diff_text = f"▲ +{delta_q:.1%} Improvement"
                diff_col = "#27ae60" # Green
            else:
                diff_text = f"▼ {delta_q:.1%} Regression"
                diff_col = "#c0392b" # Red
            
            # Show big delta in top right
            ax_plot.text(0.95, 0.95, diff_text, transform=ax_plot.transAxes, 
                         fontsize=24, fontweight='bold', color=diff_col, ha='right', va='top',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=diff_col, lw=2, alpha=0.9))

        # === Data Bars ===
        ax_text.axis('off')
        def draw_bar(y, label, val, col):
            ax_text.text(0.1, y, label, fontsize=12, va='center', ha='right', color='#555')
            ax_text.add_patch(plt.Rectangle((0.15, y-0.1), 0.7, 0.2, color='#eee', transform=ax_text.transAxes))
            ax_text.add_patch(plt.Rectangle((0.15, y-0.1), 0.7*min(val,1), 0.2, color=col, transform=ax_text.transAxes))
            ax_text.text(0.87, y, f"{val:.1%}", fontsize=12, va='center', fontweight='bold', color='#333')

        draw_bar(0.8, "Depth (Force):", c, '#3498db')
        draw_bar(0.5, "Stability (Conv):", s, '#9b59b6')
        draw_bar(0.2, "MAP Quality:", q, '#34495e')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        import PIL.Image
        return torch.from_numpy(np.array(PIL.Image.open(buf).convert("RGB")).astype(np.float32) / 255.0)[None,]

    def create_empty_image(self):
        return torch.zeros((1, 512, 512, 3))