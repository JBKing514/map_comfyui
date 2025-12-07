import torch
import numpy as np
import io
import matplotlib.pyplot as plt
import comfy.sample
import comfy.model_management
import nodes
import copy
import csv
import os
import folder_paths
from datetime import datetime

# === Global Variables ===
GLOBAL_PROJ_MATRIX = None
GLOBAL_PROJ_DIM = 0

class MAP_Pro_Suite:
    """
    The Manifold Alignment Protocol (MAP) Suite for ComfyUI.
    """
    
    # Cache for tracking improvements
    HISTORY_CACHE = {}      # Stores {seed: q_score} for exact matches
    LAST_RUN_Q = None       # Stores the absolute last Q-score (regardless of seed)
    LAST_RUN_SEED = None    # Stores the absolute last Seed

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                
                # Standard Sampler Settings
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # Operation Mode Selection
                "operation_mode": (["Analyze (Manual)", "Auto-Tune (Hill Climb)"], ),
                
                # --- Manual Analysis Params ---
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                
                # --- Auto-Tuner Params ---
                "tuner_max_steps": ("INT", {"default": 50, "min": 20, "max": 200, "tooltip": "Stop searching if steps exceed this limit."}),
                "tuner_stride": ("INT", {"default": 5, "min": 1, "max": 20, "tooltip": "Increment steps by this amount per iteration."}),
                "tuner_optimize_scheduler": ("BOOLEAN", {"default": False, "tooltip": "After finding best steps, try ALL schedulers."}),
                "tuner_refine_cfg": ("BOOLEAN", {"default": True, "tooltip": "Fine-tune CFG scale around the optimal point."}),
                "tuner_cfg_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Search range for CFG."}),
                
                # Logging
                "save_log_csv": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "optional_positive_text": ("STRING", {"forceInput": True}),
                "optional_negative_text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "analysis_plot", "report_text")
    FUNCTION = "execute_map"
    CATEGORY = "MAP_Protocol"

    def execute_map(self, model, seed, positive, negative, latent_image, sampler_name, scheduler, denoise,
                    operation_mode, steps, cfg, tuner_max_steps, tuner_stride, 
                    tuner_optimize_scheduler, tuner_refine_cfg, tuner_cfg_range,
                    save_log_csv, optional_positive_text=None, optional_negative_text=None):
        
        seed_int = int(seed)

        if operation_mode == "Analyze (Manual)":
            return self.run_manual_analysis(
                model, seed_int, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
            )
        else:
            return self.run_auto_tuner(
                model, seed_int, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                tuner_max_steps, tuner_stride, tuner_optimize_scheduler, tuner_refine_cfg, tuner_cfg_range,
                save_log_csv, optional_positive_text, optional_negative_text
            )

    # ============================================================
    # MODE A: Manual Analysis
    # ============================================================
    def run_manual_analysis(self, model, seed, steps, cfg, sampler, scheduler, pos, neg, lat, denoise):
        print(f"[MAP] Manual Run initiated (Seed={seed})...")
        
        result_lat, trajectory = self.core_sampling(model, seed, steps, cfg, sampler, scheduler, pos, neg, lat, denoise)
        
        if len(trajectory) < 5: 
            return (result_lat, self.create_empty_image(), "Error: Insufficient steps.")
        
        coords, c_norm, s_good, q_score, c_raw, ratio = self.calculate_metrics(trajectory)
        
        # === Logic Update: Smart Comparison ===
        delta_q = 0.0
        status_code = 0 
        # status_code meaning:
        # 0 = No history (First Run / Baseline) -> Show Yellow Box
        # 1 = Exact Seed Match (Valid Delta)    -> Show Solid Box
        # 2 = Different Seed (Approximation)    -> Show Dashed Box
        
        if seed in self.HISTORY_CACHE:
            # Case 1: Exact seed found in history (Best case)
            delta_q = q_score - self.HISTORY_CACHE[seed]
            status_code = 1
            print(f"[MAP] Exact Cache Hit. Delta={delta_q:+.2%}")
            
        elif self.LAST_RUN_Q is not None:
            # Case 2: No exact match, but we have a previous run (Random -> Fix Scenario)
            delta_q = q_score - self.LAST_RUN_Q
            status_code = 2
            print(f"[MAP] Seed Mismatch Comparison. Delta={delta_q:+.2%}")
            
        else:
            # Case 3: Cold Start
            status_code = 0
            print(f"[MAP] Baseline Established.")

        # Update Caches
        self.HISTORY_CACHE[seed] = q_score
        self.LAST_RUN_Q = q_score
        self.LAST_RUN_SEED = seed

        # Report Text
        status_str = " (Ref)"
        if status_code == 1: status_str = f" ({delta_q:+.2%})"
        if status_code == 2: status_str = f" ({delta_q:+.2%}*)" # * denotes diff seed

        report = (f"MAP Analysis Report:\n"
                  f"Q-Score: {q_score:.2%}{status_str}\n"
                  f"Sampler: {sampler} | Scheduler: {scheduler}\n"
                  f"CFG: {cfg:.2f} | Steps: {steps}\n"
                  f"Depth: {c_norm:.1%} | Stability: {s_good:.1%}")
                  
        plot = self.plot_manual(coords, c_norm, s_good, q_score, delta_q, status_code, cfg, scheduler)
        return (result_lat, plot, report)

    # ============================================================
    # MODE B: Auto-Tuner
    # ============================================================
    def run_auto_tuner(self, model, seed, start_steps, base_cfg, sampler, start_scheduler, pos, neg, lat, denoise,
                       max_steps, stride, optimize_scheduler, refine_cfg, cfg_range, save_csv, pos_text, neg_text):
        
        print(f"\n[MAP Auto-Tune] Starting Optimization Strategy (Seed={seed})")
        
        history = []
        best_record = None
        
        current_steps = start_steps
        current_cfg = base_cfg
        current_scheduler = start_scheduler
        
        # Phase 1: Step Search
        print(f"[Phase 1] Searching for Optimal Steps (Scheduler: {current_scheduler})...")
        while current_steps <= max_steps:
            print(f"  > Testing Steps={current_steps}...", end="", flush=True)
            result_lat, trajectory = self.core_sampling(model, seed, current_steps, current_cfg, sampler, current_scheduler, pos, neg, lat, denoise)
            
            if len(trajectory) >= 5:
                _, _, _, q_score, _, _ = self.calculate_metrics(trajectory)
                print(f" Q={q_score:.2%}")
                
                record = {'phase': 'step_search', 'steps': current_steps, 'cfg': current_cfg, 'scheduler': current_scheduler, 'q': q_score, 'lat': result_lat}
                history.append(record)
                
                if best_record is None: best_record = record
                elif q_score >= best_record['q'] * 0.995: best_record = record
                else: break 
            else: print(" Failed.")
            
            current_steps += stride
            comfy.model_management.throw_exception_if_processing_interrupted()

        # Phase 2: Scheduler Sweep
        if optimize_scheduler and best_record:
            print(f"\n[Phase 2] Sweeping Schedulers...")
            schedulers_to_try = [s for s in comfy.samplers.KSampler.SCHEDULERS if s != best_record['scheduler']]
            opt_steps = best_record['steps']
            opt_cfg = best_record['cfg']
            
            for test_sch in schedulers_to_try:
                print(f"  > Testing: {test_sch}...", end="", flush=True)
                result_lat, trajectory = self.core_sampling(model, seed, opt_steps, opt_cfg, sampler, test_sch, pos, neg, lat, denoise)
                if len(trajectory) >= 5:
                    _, _, _, q_score, _, _ = self.calculate_metrics(trajectory)
                    print(f" Q={q_score:.2%}")
                    record = {'phase': 'sch_opt', 'steps': opt_steps, 'cfg': opt_cfg, 'scheduler': test_sch, 'q': q_score, 'lat': result_lat}
                    history.append(record)
                    if q_score > best_record['q']: best_record = record
                else: print(" Failed.")

        # Phase 3: CFG Refinement
        if refine_cfg and best_record:
            print(f"\n[Phase 3] Refining CFG...")
            opt_steps = best_record['steps']
            opt_sch = best_record['scheduler']
            base = best_record['cfg']
            cfg_candidates = [base - cfg_range, base + cfg_range]
            
            for test_cfg in cfg_candidates:
                if test_cfg <= 0: continue
                print(f"  > Testing CFG={test_cfg:.2f}...", end="", flush=True)
                result_lat, trajectory = self.core_sampling(model, seed, opt_steps, test_cfg, sampler, opt_sch, pos, neg, lat, denoise)
                if len(trajectory) >= 5:
                    _, _, _, q_score, _, _ = self.calculate_metrics(trajectory)
                    print(f" Q={q_score:.2%}")
                    record = {'phase': 'cfg_refine', 'steps': opt_steps, 'cfg': test_cfg, 'scheduler': opt_sch, 'q': q_score, 'lat': result_lat}
                    history.append(record)
                    if q_score > best_record['q']: best_record = record

        if best_record:
            self.HISTORY_CACHE[seed] = best_record['q']
            # Also update Last Run for Manual continuity
            self.LAST_RUN_Q = best_record['q']
            self.LAST_RUN_SEED = seed
            if save_csv: self.save_to_csv(seed, best_record, pos_text, neg_text, sampler)

        report = (f"MAP Auto-Tune Complete:\nSeed: {seed}\nBest Q: {best_record['q']:.2%}\n"
                  f"Steps: {best_record['steps']} | CFG: {best_record['cfg']:.2f}\nSch: {best_record['scheduler']}")
        
        plot = self.plot_tuning_curve(history, best_record)
        return (best_record['lat'], plot, report)

    # === Utilities ===
    def save_to_csv(self, seed, record, pos_text, neg_text, sampler):
        try:
            output_dir = folder_paths.get_output_directory()
            csv_path = os.path.join(output_dir, "MAP_Tuning_Log.csv")
            file_exists = os.path.isfile(csv_path)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pos_content = (pos_text if pos_text else "N/A")[:200].replace("\n", " ")
            row = [timestamp, str(seed), f"{record['q']:.4f}", str(record['steps']), 
                   f"{record['cfg']:.2f}", sampler, record['scheduler'], pos_content]
            with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists: writer.writerow(["Timestamp", "Seed", "Q_Score", "Best_Steps", "Best_CFG", "Sampler", "Best_Scheduler", "Prompt"])
                writer.writerow(row)
        except Exception: pass

    def core_sampling(self, model, seed, steps, cfg, sampler, scheduler, pos, neg, lat, denoise):
        traj = []
        def cb(s, x0, x, t):
            try: traj.append(x[0].detach().cpu().flatten())
            except: pass
        orig = comfy.sample.sample
        comfy.sample.sample = lambda *a, **k: orig(*a, **{**k, 'callback': cb})
        try:
            res = nodes.common_ksampler(model, seed, steps, cfg, sampler, scheduler, pos, neg, lat, denoise)
            return res[0], traj
        except Exception as e:
            print(f"[MAP] Sampling Failed: {e}")
            return lat, []
        finally:
            comfy.sample.sample = orig

    def calculate_metrics(self, traj):
        X = torch.stack(traj).float()
        global GLOBAL_PROJ_MATRIX, GLOBAL_PROJ_DIM
        if GLOBAL_PROJ_MATRIX is None or GLOBAL_PROJ_DIM != X.shape[1]:
            P = torch.randn(X.shape[1], 2)
            GLOBAL_PROJ_MATRIX = P / torch.norm(P, dim=0, keepdim=True)
            GLOBAL_PROJ_DIM = X.shape[1]
        coords = torch.mm(X - X[0], GLOBAL_PROJ_MATRIX).numpy()
        c_raw = np.max(np.linalg.norm(coords, axis=1))
        c_norm = c_raw / (c_raw + 5.0)
        vels = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        tail_len = max(3, int(len(vels)*0.2))
        ratio = np.mean(vels[-tail_len:]) / (np.mean(vels)+1e-8)
        s_good = np.exp(-5.0 * ratio)
        return coords, c_norm, s_good, c_norm*s_good, c_raw, ratio

    # === Plotting: Manual Mode (Fixed) ===
    def plot_manual(self, coords, c, s, q, delta_q, status_code, cfg, scheduler):
        plt.figure(figsize=(10, 12), dpi=100)
        gs = plt.GridSpec(5, 1)
        ax_plot = plt.subplot(gs[0:4, 0])
        ax_text = plt.subplot(gs[4, 0])
        
        x, y = coords[:, 0], coords[:, 1]
        ax_plot.plot(x, y, color='gray', alpha=0.5, linestyle='--')
        ax_plot.scatter(x, y, c=np.arange(len(x)), cmap='coolwarm', s=100, edgecolors='black', zorder=10)
        ax_plot.text(x[0], y[0], ' Start', va='bottom')
        ax_plot.text(x[-1], y[-1], ' End', va='top', fontweight='bold')
        ax_plot.axis('equal')
        ax_plot.grid(True, alpha=0.3)
        ax_plot.set_title(f"MAP Analysis (Q={q:.1%})\nCFG: {cfg:.1f} | Sch: {scheduler}", fontsize=14, fontweight='bold', color='#333')
        
        # === Status Badge Logic ===
        if status_code == 0:
            # YELLOW BOX: Baseline Established
            ax_plot.text(0.95, 0.95, "Reference\nEstablished", transform=ax_plot.transAxes, fontsize=16, fontweight='bold', 
                         color='#7f8c8d', ha='right', va='top', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="#f1c40f", ec="gray", lw=1, alpha=0.9))
        else:
            # DELTA BOX: Improvement / Regression
            color = "#27ae60" if delta_q > 0 else "#c0392b"
            sign = "+" if delta_q > 0 else ""
            txt = f"{sign}{delta_q:.1%} Change"
            
            # Line Style: Solid for Exact Seed, Dashed for Diff Seed
            edge_style = 'solid' if status_code == 1 else 'dashed'
            if status_code == 2:
                txt += "\n(Diff Seed)"
            
            ax_plot.text(0.95, 0.95, txt, transform=ax_plot.transAxes, fontsize=24, fontweight='bold', 
                         color=color, ha='right', va='top', 
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=2, linestyle=edge_style, alpha=0.9))
        
        ax_text.axis('off')
        self._draw_bar(ax_text, 0.8, "Depth:", c, '#3498db')
        self._draw_bar(ax_text, 0.5, "Stability:", s, '#9b59b6')
        self._draw_bar(ax_text, 0.2, "Quality:", q, '#34495e')
        return self._plt_to_tensor()

    # === Plotting: Auto-Tuner ===
    def plot_tuning_curve(self, history, best):
        plt.figure(figsize=(10, 8), dpi=100)
        gs = plt.GridSpec(6, 1)
        ax_plot = plt.subplot(gs[0:5, 0])
        ax_info = plt.subplot(gs[5, 0])
        
        step_data = [h for h in history if h['phase'] == 'step_search']
        sch_data = [h for h in history if h['phase'] == 'sch_opt']
        cfg_data = [h for h in history if h['phase'] == 'cfg_refine']
        
        ax_plot.plot([h['steps'] for h in step_data], [h['q'] for h in step_data], 
                 marker='o', linestyle='-', color='#3498db', linewidth=2, label="Step Search")
        if sch_data:
            ax_plot.scatter([h['steps'] for h in sch_data], [h['q'] for h in sch_data], 
                        marker='^', s=100, color='#9b59b6', alpha=0.7, label="Scheduler Try", zorder=5)
        if cfg_data:
            ax_plot.scatter([h['steps'] for h in cfg_data], [h['q'] for h in cfg_data], 
                        marker='*', s=150, color='#e67e22', alpha=0.8, label="CFG Try", zorder=6)
        ax_plot.scatter(best['steps'], best['q'], s=250, c='#2ecc71', edgecolors='black', linewidth=2, zorder=10, label="Global Best")
        ax_plot.annotate('Best', xy=(best['steps'], best['q']), xytext=(0, 15), 
                         textcoords='offset points', ha='center', va='bottom', color='#27ae60', fontweight='bold', 
                         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='#27ae60'))
        
        all_qs = [h['q'] for h in history]
        y_min, y_max = min(all_qs), max(all_qs)
        margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.05
        ax_plot.set_ylim(bottom=y_min - margin, top=y_max + margin)
        ax_plot.set_title("MAP Auto-Tuner Optimization Path", fontsize=14, fontweight='bold')
        ax_plot.set_xlabel("Sampling Steps")
        ax_plot.set_ylabel("MAP Quality Score")
        ax_plot.legend(loc='best', framealpha=0.8)
        ax_plot.grid(True, alpha=0.3)
        
        ax_info.axis('off')
        ax_info.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_info.transAxes, color='#f8f9fa', zorder=0))
        ax_info.text(0.05, 0.5, "OPTIMAL\nFOUND", transform=ax_info.transAxes, va='center', ha='left', fontsize=12, fontweight='bold', color='#2c3e50')
        ax_info.text(0.18, 0.5, f"{best['q']:.2%}", transform=ax_info.transAxes, va='center', ha='left', fontsize=22, fontweight='bold', color='#27ae60')
        ax_info.text(0.18, 0.2, "Quality Score", transform=ax_info.transAxes, va='center', ha='left', fontsize=8, color='#7f8c8d')
        ax_info.plot([0.33, 0.33], [0.2, 0.8], transform=ax_info.transAxes, color='#bdc3c7', linewidth=1)
        self._draw_info_item(ax_info, 0.38, 0.5, "Best Steps", str(best['steps']))
        self._draw_info_item(ax_info, 0.52, 0.5, "Best CFG", f"{best['cfg']:.2f}")
        ax_info.plot([0.64, 0.64], [0.2, 0.8], transform=ax_info.transAxes, color='#bdc3c7', linewidth=1)
        ax_info.text(0.69, 0.65, "Best Scheduler", transform=ax_info.transAxes, va='center', ha='left', fontsize=9, color='#7f8c8d')
        sch_name = best['scheduler']
        font_size = 14 if len(sch_name) < 15 else 10
        ax_info.text(0.69, 0.35, sch_name, transform=ax_info.transAxes, va='center', ha='left', fontsize=font_size, fontweight='bold', color='#2c3e50')
        plt.tight_layout()
        return self._plt_to_tensor()

    def _draw_info_item(self, ax, x, y, label, value):
        ax.text(x, y + 0.12, label, transform=ax.transAxes, fontsize=9, color='#7f8c8d', va='bottom', ha='left')
        ax.text(x, y - 0.08, value, transform=ax.transAxes, fontsize=16, fontweight='bold', color='#2c3e50', va='top', ha='left')

    def _draw_bar(self, ax, y, label, val, col):
        ax.text(0.1, y, label, fontsize=12, va='center', ha='right', color='#555')
        ax.add_patch(plt.Rectangle((0.15, y-0.1), 0.7, 0.2, color='#eee', transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((0.15, y-0.1), 0.7*min(val,1), 0.2, color=col, transform=ax.transAxes))
        ax.text(0.87, y, f"{val:.1%}", fontsize=12, va='center', fontweight='bold', color='#333')

    def _plt_to_tensor(self):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        import PIL.Image
        return torch.from_numpy(np.array(PIL.Image.open(buf).convert("RGB")).astype(np.float32) / 255.0)[None,]

    def create_empty_image(self): return torch.zeros((1, 512, 512, 3))