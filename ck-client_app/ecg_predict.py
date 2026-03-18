import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import random
import numpy as np

# Matplotlib imports for embedding the graph directly in Tkinter
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import strictly the processing and classification functions from your core module
from ecg_core import preprocess_ecg, segment_beats, authenticate_users


class CardioKeyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CardioKey : ECG Biometric Identification")
        self.root.geometry("950x650")

        # Apply modern native theme
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        # State Variables
        self.data_dir = "ecgdb"
        self.live_file = ""
        self.clf = None
        self.pca = None

        # --- UI LAYOUT ---
        main_frame = ttk.Frame(root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL (Controls) ---
        left_frame = ttk.Frame(main_frame, width=280)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        # Model Training Section
        ttk.Label(left_frame, text="Model Training", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 2))
        self.lbl_db = ttk.Label(left_frame, text=f"Database: {self.data_dir}", foreground="gray")
        self.lbl_db.pack(anchor="w", pady=(0, 10))

        ttk.Button(left_frame, text="Select Database", command=self.select_db).pack(fill=tk.X, pady=2)
        self.btn_train = ttk.Button(left_frame, text="Train Classifier", command=self.train_system)
        self.btn_train.pack(fill=tk.X, pady=(2, 20))

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Prediction Section
        ttk.Label(left_frame, text="Live Subject Prediction", font=("Helvetica", 12, "bold")).pack(anchor="w",
                                                                                                   pady=(0, 2))
        self.lbl_live = ttk.Label(left_frame, text="Live File: [Auto-Demo Fallback]", foreground="gray")
        self.lbl_live.pack(anchor="w", pady=(0, 10))

        ttk.Button(left_frame, text="Upload ECG (.txt)", command=self.select_live_file).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="Predict & Visualize", command=self.predict_subject).pack(fill=tk.X, pady=2)

        # Output Log (Moved to bottom left for a cleaner look)
        ttk.Label(left_frame, text="System Output:", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(20, 2))
        self.txt_output = tk.Text(left_frame, width=35, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9),
                                  relief=tk.FLAT)
        self.txt_output.pack(fill=tk.BOTH, expand=True)
        self.log_output("[*] GUI Initialized.\n[*] Waiting for training...")

        # --- RIGHT PANEL (Visualization) ---
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Embedded Matplotlib Figure
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # Top plot (Raw vs Filtered)
        self.ax2 = self.fig.add_subplot(212)  # Bottom plot (Segmented Beats)
        self.setup_plot_style()

        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def setup_plot_style(self):
        """Initializes empty graphs with styling"""
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.set_title("Stage 1 & 2: Signal Conditioning", fontsize=10, fontweight='bold')
        self.ax1.set_ylabel("Amplitude")
        self.ax1.grid(True, linestyle='--', alpha=0.6)

        self.ax2.set_title("Stage 3: Extracted P-QRS-T Fragments (150pt)", fontsize=10, fontweight='bold')
        self.ax2.set_xlabel("Samples (300 Hz)")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.grid(True, linestyle='--', alpha=0.6)
        self.fig.tight_layout()

    def update_visualization(self, raw, clean, beats, predicted_id):
        """Updates the embedded Matplotlib canvas with new signal data"""
        self.setup_plot_style()

        # Plot 1: Raw vs Clean (First 5 seconds / 1500 samples)
        self.ax1.plot(raw[:1500], label='Raw Hardware Signal', alpha=0.5, color='gray')
        self.ax1.plot(clean[:1500], label='Filtered Signal', color='red')
        self.ax1.set_title(f"Signal Conditioning (Predicted Subject: {predicted_id})", fontsize=10, fontweight='bold')
        self.ax1.legend(loc="upper right", fontsize=8)

        # Plot 2: Extracted Beats
        for b in beats:
            self.ax2.plot(b, alpha=0.3, color='blue')
        if len(beats) > 0:
            self.ax2.plot(np.mean(beats, axis=0), color='black', linewidth=2, label='Mean Biometric Signature')
        self.ax2.legend(loc="upper right", fontsize=8)

        self.fig.tight_layout()
        self.canvas.draw()

    def log_output(self, message):
        """Prints text to the GUI terminal"""
        self.txt_output.insert(tk.END, message + "\n")
        self.txt_output.see(tk.END)
        self.root.update()

    def select_db(self):
        folder = filedialog.askdirectory(title="Select ECG Database")
        if folder:
            self.data_dir = folder
            self.lbl_db.config(text=f"Database: {os.path.basename(self.data_dir)}")
            self.log_output(f"\n[*] DB set to: {self.data_dir}")

    def select_live_file(self):
        file = filedialog.askopenfilename(title="Select Live ECG Sample", filetypes=[("Text Files", "*.txt")])
        if file:
            self.live_file = file
            self.lbl_live.config(text=f"Live File: {os.path.basename(self.live_file)}")
            self.log_output(f"\n[*] Loaded sample: {os.path.basename(self.live_file)}")

    def train_system(self):
        if not os.path.exists(self.data_dir):
            messagebox.showerror("Error", "Database folder not found!")
            return

        self.log_output("\n[*] Compiling Dataset...")
        self.btn_train.config(state=tk.DISABLED)

        dataset_X_rest, dataset_Y_rest = [], []

        for i in range(1, 91):
            filepath = os.path.join(self.data_dir, f"{i}.txt")
            if os.path.exists(filepath):
                raw_signal = np.loadtxt(filepath)
                subject_id = (i + 1) // 2
                is_exercise = (i % 2 == 0)

                clean_signal = preprocess_ecg(raw_signal, fs=300)
                beats = segment_beats(clean_signal, fs=300, num_beats=30)

                for beat in beats:
                    if not is_exercise:
                        dataset_X_rest.append(beat)
                        dataset_Y_rest.append(subject_id)

        dataset_X_rest, dataset_Y_rest = np.array(dataset_X_rest), np.array(dataset_Y_rest)

        if len(dataset_X_rest) > 0:
            self.log_output("[*] Training Models...")
            self.clf, self.pca = authenticate_users(dataset_X_rest, dataset_Y_rest)
            self.btn_train.config(text="Models Trained")
            self.log_output("[+] System Ready.")
        else:
            self.log_output("[-] Error: No data.")
            self.btn_train.config(state=tk.NORMAL)

    def predict_subject(self):
        if self.clf is None or self.pca is None:
            messagebox.showwarning("Warning", "Train the system first!")
            return

        self.log_output("\n--- PREDICTION ATTEMPT ---")

        is_demo = False
        true_id = None

        if not self.live_file or not os.path.exists(self.live_file):
            self.log_output("[!] Auto-Demo Mode...")
            is_demo = True

            rand_idx = random.choice(range(1, 91, 2))
            true_id = (rand_idx + 1) // 2
            sim_path = os.path.join(self.data_dir, f"{rand_idx}.txt")

            full_sig = np.loadtxt(sim_path)
            self.live_file = "temp_live_test.txt"
            np.savetxt(self.live_file, full_sig[-3000:], fmt="%.6f")

        self.log_output(f"[*] Analyzing signal...")
        live_raw = np.loadtxt(self.live_file)

        live_clean = preprocess_ecg(live_raw, fs=300)
        live_beats = segment_beats(live_clean, fs=300, num_beats=1)

        if len(live_beats) > 0:
            live_30pt = self.pca.transform(live_beats.reshape(1, -1))

            predicted_id = self.clf.predict(live_30pt)
            probabilities = self.clf.predict_proba(live_30pt)
            confidence = np.max(probabilities) * 100

            self.log_output(f"\n > Predicted ID: Subj {predicted_id}")
            self.log_output(f" > Confidence:   {confidence:.2f}%")

            if is_demo:
                self.log_output(f" > Ground Truth: Subj {true_id}")
                if predicted_id == true_id:
                    self.log_output(" > Eval: CORRECT")
                else:
                    self.log_output(" > Eval: INCORRECT")

            self.log_output("[*] Updating Graphs...")
            self.update_visualization(live_raw, live_clean, live_beats, predicted_id)

        else:
            self.log_output("[-] Error: No heartbeat.")


if __name__ == "__main__":
    root = tk.Tk()
    app = CardioKeyGUI(root)
    root.mainloop()
