"""
cpu_bench_gui_v2.py

Enhanced CPU Benchmark Analyzer (Tkinter) with:
 - live matplotlib visualizations
 - loading animation + spinner
 - cancel button
 - repeat-runs (median & stddev)
 - history persistence and compare-plot
 - everything runs locally on your machine

Usage:
    python cpu_bench_gui_v2.py

Requirements:
    pip install psutil py-cpuinfo numpy matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import json
import os
from datetime import datetime
import platform
import psutil
import cpuinfo
import numpy as np
import hashlib
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import statistics

# ----------------------
# Config
# ----------------------
HISTORY_FILE = "bench_history.json"
DEFAULT_MATRIX_N = 300
MAX_WORKERS = max(1, psutil.cpu_count(logical=True))

# ----------------------
# System info helper
# ----------------------
def get_system_info():
    info = cpuinfo.get_cpu_info()
    try:
        freq = psutil.cpu_freq().current
        freq_str = f"{freq:.2f} MHz"
    except Exception:
        freq_str = "N/A"
    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "processor": info.get("brand_raw", platform.processor()),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "frequency": freq_str,
    }

# ----------------------
# Benchmark implementations
# Each returns a numeric result; for bandwidth tests returns tuple (time, gbps) etc.
# ----------------------
def integer_arithmetic_test(iterations=5_000_000):
    start = time.perf_counter()
    x = 0
    for i in range(iterations):
        x += (i * 3) ^ (i >> 1)
        x -= i % 7
    return time.perf_counter() - start

def float_arithmetic_test(iterations=2_000_000):
    start = time.perf_counter()
    x = 0.0
    for i in range(iterations):
        x += math.sqrt((i + 1) * 0.5)
        x *= 1.0000001
    return time.perf_counter() - start

def matrix_multiply_test(n=DEFAULT_MATRIX_N):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    start = time.perf_counter()
    _ = np.dot(A, B)
    return time.perf_counter() - start

def memory_bandwidth_test(size_mb=200):
    size = int(size_mb * 1024 * 1024 / 8)
    a = np.zeros(size, dtype=np.float64)
    # write
    start = time.perf_counter()
    a[:] = np.arange(size, dtype=np.float64)
    write_time = time.perf_counter() - start
    # read
    start = time.perf_counter()
    _ = np.sum(a)
    read_time = time.perf_counter() - start
    bytes_moved = a.nbytes
    write_bw = bytes_moved / write_time / (1024 ** 3) if write_time > 0 else float('inf')
    read_bw = bytes_moved / read_time / (1024 ** 3) if read_time > 0 else float('inf')
    return {"write_time": write_time, "read_time": read_time, "write_gbps": write_bw, "read_gbps": read_bw}

def random_access_latency_test(size_mb=100, accesses=500_000):
    size = int(size_mb * 1024 * 1024 / 8)
    a = np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed=12345)
    idx = rng.integers(0, size, accesses, dtype=np.int64)
    start = time.perf_counter()
    s = 0
    for i in idx:
        s += int(a[i])
    duration = time.perf_counter() - start
    return duration

def sorting_test(n=1_000_000):
    rng = np.random.default_rng(seed=42)
    arr = rng.random(n).tolist()
    start = time.perf_counter()
    arr.sort()
    return time.perf_counter() - start

def fft_test(n=1<<18):
    x = np.random.rand(n)
    start = time.perf_counter()
    np.fft.fft(x)
    return time.perf_counter() - start

def hashing_test(iterations=120_000):
    start = time.perf_counter()
    s = b"cpu-bench"
    for i in range(iterations):
        hashlib.sha256(s + i.to_bytes(4, 'little')).digest()
    return time.perf_counter() - start

def threading_micro_test(work_items=10000):
    def tiny_work(n):
        x = 0
        for i in range(n):
            x += i & 3
        return x
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(8, MAX_WORKERS)) as ex:
        futures = [ex.submit(tiny_work, 100) for _ in range(int(work_items/100))]
        for f in futures:
            _ = f.result()
    return time.perf_counter() - start

# mapping keys -> callable + human name
BENCHMARKS = {
    "int": ("Integer Arithmetic", integer_arithmetic_test),
    "float": ("Float Arithmetic (FLOPS-ish)", float_arithmetic_test),
    "matrix": ("Matrix Multiply (NumPy)", matrix_multiply_test),
    "mem_bw": ("Memory Bandwidth (seq R/W)", memory_bandwidth_test),
    "rand_mem": ("Random Memory Access", random_access_latency_test),
    "sort": ("Sorting", sorting_test),
    "fft": ("FFT (NumPy)", fft_test),
    "hash": ("SHA256 Hashing", hashing_test),
    "threading": ("Threading micro-test", threading_micro_test),
}

# ----------------------
# GUI App
# ----------------------
class CPUBenchApp:
    def __init__(self, root):
        self.root = root
        root.title("⚙ CPU Benchmark Analyzer — Enhanced")
        root.geometry("1080x720")
        style = ttk.Style(root)
        try:
            style.theme_use('vista')
        except Exception:
            style.theme_use('clam')
        style.configure('TButton', padding=6)
        self.mainframe = ttk.Frame(root, padding=10)
        self.mainframe.pack(fill=tk.BOTH, expand=True)

        # Top: title & system info
        self._build_top()

        # Left: control panel
        self._build_left_controls()

        # Right: visualization
        self._build_visual_area()

        # Bottom: log
        self._build_log_area()

        # state
        self.history = self._load_history()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._running = False
        self._cancel_requested = False
        # spinner state
        self._spinner_chars = ['⣾','⣽','⣻','⢿','⡿','⣟','⣯','⣷']
        self._spinner_idx = 0

    def _build_top(self):
        top = ttk.Frame(self.mainframe)
        top.pack(fill=tk.X)
        ttk.Label(top, text="⚙ CPU Benchmark Analyzer — Enhanced", font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT)
        ttk.Label(top, text="Local tests • Save & compare • Live plots", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=8)
        sys_frame = ttk.LabelFrame(self.mainframe, text="System Info", padding=8)
        sys_frame.pack(fill=tk.X, pady=(8,10))
        info = get_system_info()
        col = 0
        for k, v in info.items():
            ttk.Label(sys_frame, text=f"{k.replace('_',' ').title()}:").grid(row=0, column=col*2, sticky=tk.W, padx=6)
            ttk.Label(sys_frame, text=str(v)).grid(row=0, column=col*2+1, sticky=tk.W, padx=6)
            col += 1

    def _build_left_controls(self):
        left = ttk.Frame(self.mainframe)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        # user profile
        ttk.Label(left, text="User Profile:").pack(anchor=tk.W)
        self.user_entry = ttk.Entry(left, width=28)
        self.user_entry.pack(anchor=tk.W, pady=(0,6))
        try:
            self.user_entry.insert(0, os.getlogin())
        except Exception:
            self.user_entry.insert(0, "User")

        # test buttons
        ttk.Button(left, text="Run Selected Tests", command=self.run_selected).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Run All Benchmarks", command=self.run_all).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Cancel Running Test", command=self.request_cancel).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Export Last Result", command=self.export_last_result).pack(fill=tk.X, pady=4)
        ttk.Button(left, text="Show History Comparison", command=self.show_history_plot).pack(fill=tk.X, pady=4)

        # test selection
        tests_frame = ttk.LabelFrame(left, text="Select Benchmarks", padding=8)
        tests_frame.pack(fill=tk.X, pady=(8,0))
        self.test_vars = {}
        for key, (label, _) in BENCHMARKS.items():
            v = tk.BooleanVar(value=True if key in ("matrix","mem_bw","int","float") else False)
            cb = ttk.Checkbutton(tests_frame, text=label, variable=v)
            cb.pack(anchor=tk.W)
            self.test_vars[key] = v

        # advanced options
        opts = ttk.LabelFrame(left, text="Options", padding=8)
        opts.pack(fill=tk.X, pady=(8,0))
        ttk.Label(opts, text="Matrix size (n x n):").grid(row=0, column=0, sticky=tk.W)
        self.matrix_size = tk.IntVar(value=DEFAULT_MATRIX_N)
        ttk.Entry(opts, textvariable=self.matrix_size, width=8).grid(row=0, column=1, sticky=tk.W, padx=4)
        ttk.Label(opts, text="Memory test MB:").grid(row=1, column=0, sticky=tk.W)
        self.mem_mb = tk.IntVar(value=200)
        ttk.Entry(opts, textvariable=self.mem_mb, width=8).grid(row=1, column=1, sticky=tk.W, padx=4)
        ttk.Label(opts, text="Repeat each test N times:").grid(row=2, column=0, sticky=tk.W)
        self.repeat_n = tk.IntVar(value=3)
        ttk.Entry(opts, textvariable=self.repeat_n, width=8).grid(row=2, column=1, sticky=tk.W, padx=4)

        # progress & spinner
        self.progress = ttk.Progressbar(left, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(8,4))
        spinner_frame = ttk.Frame(left)
        spinner_frame.pack(fill=tk.X)
        self.spinner_label = ttk.Label(spinner_frame, text="⣾", font=("Segoe UI", 14))
        self.spinner_label.pack(side=tk.LEFT, padx=(0,6))
        self.status_label = ttk.Label(spinner_frame, text="Idle")
        self.status_label.pack(side=tk.LEFT)

    def _build_visual_area(self):
        right = ttk.Frame(self.mainframe)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        plot_frame = ttk.LabelFrame(right, text="Live Visualization", padding=8)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.fig, self.ax = plt.subplots(figsize=(7,5))
        self.ax.set_title("Benchmark results")
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.clear_plot()

    def _build_log_area(self):
        log_frame = ttk.LabelFrame(self.mainframe, text="Results & Logs", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8,0))
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ----------------------
    # UI Utilities
    # ----------------------
    def log(self, s):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] {s}\n")
        self.log_text.see(tk.END)

    def clear_plot(self):
        self.ax.cla()
        self.ax.set_title("Benchmark results")
        self.canvas.draw()

    def _set_ui_running(self, running=True):
        self._running = running
        if running:
            self.status_label.config(text="Running tests...")
            self.progress.start(10)  # indeterminate movement
            self._animate_spinner()
        else:
            self.status_label.config(text="Idle")
            self.progress.stop()

    def _animate_spinner(self):
        # spinner animation while running
        if not self._running:
            self.spinner_label.config(text='')
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
        self.spinner_label.config(text=self._spinner_chars[self._spinner_idx])
        self.root.after(120, self._animate_spinner)

    def request_cancel(self):
        if self._running:
            self._cancel_requested = True
            self.log("Cancel requested — will stop after current test finishes.")
        else:
            messagebox.showinfo("Not running", "No benchmark is currently running.")

    # ----------------------
    # Running benchmarks
    # ----------------------
    def run_all(self):
        for k in self.test_vars:
            self.test_vars[k].set(True)
        self.run_selected()

    def run_selected(self):
        if self._running:
            messagebox.showwarning("Already running", "Please wait until the current run finishes or cancel it.")
            return
        selected = [k for k, v in self.test_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("No tests", "Select at least one benchmark.")
            return
        user = self.user_entry.get().strip() or "User"
        repeat = max(1, int(self.repeat_n.get()))
        # prepare UI
        self._cancel_requested = False
        self._set_ui_running(True)
        self.progress['mode'] = 'indeterminate'
        self.log(f"Starting benchmarks for {user}: {', '.join(selected)} (repeat={repeat})")
        self.progress.config(mode='determinate', maximum=len(selected)*repeat, value=0)
        threading.Thread(target=self._run_sequence, args=(user, selected, repeat), daemon=True).start()

    def _run_sequence(self, user, selected, repeat):
        results_entry = {"user": user, "timestamp": datetime.now().isoformat(), "system": get_system_info(), "results": {}}
        try:
            # for plotting live
            plot_data = {}
            total_steps = len(selected) * repeat
            completed_steps = 0
            for test_key in selected:
                if self._cancel_requested:
                    self.log("Run cancelled by user.")
                    break
                label, func = BENCHMARKS[test_key]
                self.log(f"Starting {label} (repeat {repeat})")
                per_run_vals = []
                for r in range(repeat):
                    if self._cancel_requested:
                        break
                    # before-run update
                    self.root.after(0, lambda lab=label, idx=r+1, tot=repeat: self.status_label.config(text=f"Running {lab} ({idx}/{tot})"))
                    # run test (some tests need args)
                    if test_key == "matrix":
                        n = max(20, int(self.matrix_size.get()))
                        res = func(n=n)
                        keyname = f"matrix_{n}_sec"
                    elif test_key == "mem_bw":
                        mb = max(10, int(self.mem_mb.get()))
                        res = func(size_mb=mb)
                        # res is a dict
                        per_run_vals.append(res)
                        completed_steps += 1
                        self.root.after(0, lambda v=completed_steps: self.progress.config(value=v))
                        self.log(f"Mem test run {r+1}: write {res['write_time']:.3f}s ({res['write_gbps']:.2f} GB/s), read {res['read_time']:.3f}s ({res['read_gbps']:.2f} GB/s)")
                        continue
                    elif test_key == "rand_mem":
                        res = func(size_mb=min(500, max(10, int(self.mem_mb.get()))))
                    elif test_key == "sort":
                        res = func(n=500_000)
                    elif test_key == "fft":
                        res = func(n=1<<18)
                    else:
                        res = func()
                    per_run_vals.append(res)
                    completed_steps += 1
                    self.root.after(0, lambda v=completed_steps: self.progress.config(value=v))
                    self.log(f"{label} run {r+1}: {res:.3f} s" if not isinstance(res, dict) else f"{label} run {r+1}: (bandwidth)")
                    # update live plot partial
                    plot_data.setdefault(label, []).append(res)
                    self.root.after(0, lambda pd=plot_data: self._plot_partial(pd))
                # aggregate per-test results
                if per_run_vals:
                    # if memory tests returned dicts, compute medians separately
                    if isinstance(per_run_vals[0], dict):
                        # aggregate bandwidth keys
                        agg = {}
                        for k in per_run_vals[0].keys():
                            vals = [v[k] for v in per_run_vals]
                            agg[f"{k}_median"] = statistics.median(vals)
                            agg[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                        results_entry['results'][label] = agg
                    else:
                        median = statistics.median(per_run_vals)
                        std = statistics.pstdev(per_run_vals) if len(per_run_vals) > 1 else 0.0
                        results_entry['results'][label] = {"median_sec": median, "std_sec": std, "raw": per_run_vals}
                if self._cancel_requested:
                    break
            # done
            self.last_result = results_entry
            self._save_history_entry(results_entry)
            self.log("Benchmarks finished (or stopped). Results saved.")
            self.root.after(0, lambda: self._plot_final(results_entry))
        except Exception as e:
            self.log(f"Error during benchmarks: {e}")
        finally:
            self._set_ui_running(False)
            self._cancel_requested = False
            self.progress.config(value=0)

    # ----------------------
    # Plotting
    # ----------------------
    def _plot_partial(self, plot_data):
        # plot_data: {label: [vals...], ...}
        self.ax.cla()
        labels = []
        values = []
        for label, vals in plot_data.items():
            # if dict-like result (mem), skip partial
            if isinstance(vals[0], dict):
                continue
            labels.append(label)
            values.append(vals[-1])  # latest run
        if labels:
            self.ax.barh(range(len(labels)), values, align='center')
            self.ax.set_yticks(range(len(labels)))
            self.ax.set_yticklabels(labels)
            self.ax.set_xlabel('seconds (lower is better)')
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_final(self, results_entry):
        self.ax.cla()
        times = {}
        bandwidths = {}
        for label, data in results_entry.get('results', {}).items():
            if isinstance(data, dict):
                # memory bandwidth case uses keys like write_gbps_median etc
                # detect bandwidth keys
                bw_keys = {k: v for k, v in data.items() if 'gb' in k or 'gbs' in k or 'write_gbps' in k or 'read_gbps' in k}
                time_keys = {k: v for k, v in data.items() if 'sec' in k}
                # prioritize medians
                for k, v in time_keys.items():
                    times[f"{label} {k}"] = v
                for k, v in bw_keys.items():
                    bandwidths[f"{label} {k}"] = v
            else:
                times[label] = data.get('median_sec', float('nan'))
        # plot times as barh
        if times:
            names = list(times.keys())
            vals = [times[n] for n in names]
            self.ax.barh(range(len(names)), vals, align='center')
            self.ax.set_yticks(range(len(names)))
            self.ax.set_yticklabels(names)
            self.ax.set_xlabel('seconds (lower is better)')
        # plot bandwidths in twin axis
        if bandwidths:
            ax2 = self.ax.twiny()
            names2 = list(bandwidths.keys())
            vals2 = [bandwidths[n] for n in names2]
            ax2.plot(range(len(names2)), vals2, marker='o')
            ax2.set_xticks(range(len(names2)))
            ax2.set_xticklabels(names2, rotation=45, ha='right')
            ax2.set_xlabel('GB/s (higher is better)')
        self.fig.tight_layout()
        self.canvas.draw()

    # ----------------------
    # History persistence
    # ----------------------
    def _load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_history_entry(self, entry):
        hist = self.history
        hist.append(entry)
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(hist, f, indent=2)
            self.history = hist
        except Exception as e:
            self.log(f"Failed to write history: {e}")

    def export_last_result(self):
        if not hasattr(self, 'last_result'):
            messagebox.showinfo("No results", "No benchmark result available to export yet.")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files","*.json")], title="Export last result")
        if not fn:
            return
        try:
            with open(fn, 'w') as f:
                json.dump(self.last_result, f, indent=2)
            messagebox.showinfo("Exported", f"Last result exported to {fn}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export file: {e}")

    # ----------------------
    # History comparison plot (pop-up)
    # ----------------------
    def show_history_plot(self):
        if not self.history:
            messagebox.showinfo("No history", "No saved benchmark history yet.")
            return
        # build a pop-up with a matplotlib plot comparing medians across runs
        popup = tk.Toplevel(self.root)
        popup.title("History comparison")
        popup.geometry("900x600")
        fig, ax = plt.subplots(figsize=(8,5))
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # choose last N history entries
        entries = self.history[-10:]
        # collect a set of metric names across entries
        metric_set = set()
        for e in entries:
            for k in e.get('results', {}).keys():
                metric_set.add(k)
        metric_list = sorted(metric_set)
        # for each metric, show median over time
        x = list(range(len(entries)))
        for metric in metric_list:
            vals = []
            for e in entries:
                v = e.get('results', {}).get(metric)
                if isinstance(v, dict):
                    # try to pick a representative numeric field
                    if 'median_sec' in v:
                        vals.append(v['median_sec'])
                    elif 'write_gbps_median' in v:
                        vals.append(v['write_gbps_median'])
                    else:
                        vals.append(np.nan)
                else:
                    vals.append(np.nan)
            ax.plot(x, vals, marker='o', label=metric)
        ax.set_title("History comparison (last runs)")
        ax.set_xlabel("Run index (older -> newer)")
        ax.set_ylabel("seconds / (GB/s)")
        ax.legend(loc='upper left', bbox_to_anchor=(1,1))
        fig.tight_layout()
        canvas.draw()

# ----------------------
# Main
# ----------------------
def main():
    root = tk.Tk()
    app = CPUBenchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
