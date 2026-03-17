import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import time
import os
import threading
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ECGRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CardioKey : ECG Recorder (Sampling)")
        self.root.geometry("980x750")
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        self.db_dir = "ECGIDld2_db"
        self.dest_var = tk.StringVar(value="LIVE") 
        self.activity_state = tk.StringVar(value="REST") # Default to Resting (Odd)
        self.is_recording = False
        
        self.setup_ui()
        
        # Instantly load demonstration data if available (triggers calculation)
        self.load_default_demonstration()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL (Controls) ---
        left_frame = ttk.Frame(main_frame, width=320)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        # Connection Setup
        ttk.Label(left_frame, text="Connection Setup", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 5))
        port_frame = ttk.Frame(left_frame)
        port_frame.pack(fill=tk.X, pady=2)
        ttk.Label(port_frame, text="Arduino Port:").pack(side=tk.LEFT)
        
        common_ports = [
            "/dev/ttyUSB0", "/dev/ttyUSB1", 
            "/dev/ttyACM0", "/dev/ttyACM1",
            "COM3", "COM4", "COM5"
        ]
        self.combo_port = ttk.Combobox(port_frame, values=common_ports, width=15)
        self.combo_port.set("/dev/ttyUSB0") 
        self.combo_port.pack(side=tk.RIGHT)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Dynamic Firmware Configuration
        ttk.Label(left_frame, text="Hardware Configuration", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        rate_frame = ttk.Frame(left_frame)
        rate_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rate_frame, text="Sampling Rate (Hz):").pack(side=tk.LEFT)
        
        # Dropdown for valid ADS1115 sampling rates
        valid_rates = ["100", "125", "200", "250", "300", "400", "500", "800", "860"]
        self.combo_rate = ttk.Combobox(rate_frame, values=valid_rates, width=8)
        self.combo_rate.set("300") # Default to methodology standard
        self.combo_rate.pack(side=tk.RIGHT)
        
        dur_frame = ttk.Frame(left_frame)
        dur_frame.pack(fill=tk.X, pady=2)
        ttk.Label(dur_frame, text="Duration (Seconds):").pack(side=tk.LEFT)
        self.entry_duration = ttk.Entry(dur_frame, width=11)
        self.entry_duration.insert(0, "300")
        self.entry_duration.pack(side=tk.RIGHT)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Destination Selection
        ttk.Label(left_frame, text="Target Destination", font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Radiobutton(left_frame, text="Live Authentication (live_test.txt)", 
                        variable=self.dest_var, value="LIVE").pack(anchor="w", pady=2)
        ttk.Radiobutton(left_frame, text="Add to Database (Auto-numbered)", 
                        variable=self.dest_var, value="DB").pack(anchor="w", pady=2)
                        
        # Database Activity State (Odd/Even Logic)
        state_frame = ttk.LabelFrame(left_frame, text=" Database Activity State ", padding=5)
        state_frame.pack(fill=tk.X, pady=(5, 5))
        ttk.Radiobutton(state_frame, text="Resting Baseline (Saves as Odd ID)", 
                        variable=self.activity_state, value="REST").pack(anchor="w", pady=2)
        ttk.Radiobutton(state_frame, text="Post-Exercise (Saves as Even ID)", 
                        variable=self.activity_state, value="EXERCISE").pack(anchor="w", pady=2)
        
        self.lbl_db = ttk.Label(left_frame, text=f"DB Path: {self.db_dir}", foreground="gray")
        self.lbl_db.pack(anchor="w", pady=(2, 5))
        ttk.Button(left_frame, text="Change Database Folder", command=self.select_db).pack(fill=tk.X, pady=5)
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=15)
        
        # Action Buttons
        self.btn_start = ttk.Button(left_frame, text="Start Acquisition", command=self.start_recording)
        self.btn_start.pack(fill=tk.X, pady=5)
        
        # Output Log
        ttk.Label(left_frame, text="Acquisition Log", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(15, 2))
        self.txt_output = tk.Text(left_frame, width=38, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 9), relief=tk.FLAT)
        self.txt_output.pack(fill=tk.BOTH, expand=True)
        self.log_output("[*] Ready to configure & connect.")

        # --- RIGHT PANEL (Visualization & Specs) ---
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Dynamic Firmware Specifications Display
        specs_frame = ttk.LabelFrame(right_frame, text=" Active Sampling Details ", padding=10)
        specs_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.lbl_sps = ttk.Label(specs_frame, text="Sampling Rate: Pending...", font=("Helvetica", 10, "bold"))
        self.lbl_sps.grid(row=0, column=0, padx=10, sticky="w")
        
        self.lbl_time = ttk.Label(specs_frame, text="Recording Time: Pending...", font=("Helvetica", 10, "bold"))
        self.lbl_time.grid(row=0, column=1, padx=10, sticky="w")
        
        self.lbl_total = ttk.Label(specs_frame, text="Total Samples: Pending...", font=("Helvetica", 10, "bold"))
        self.lbl_total.grid(row=1, column=0, columnspan=2, padx=10, pady=(5,0), sticky="w")

        # Embedded Matplotlib Figure
        self.fig = Figure(figsize=(6, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_default_demonstration(self):
        default_file = "live_test.txt"
        if os.path.exists(default_file) and os.path.getsize(default_file) > 0:
            self.log_output(f"\n[*] Found existing data in '{default_file}'.")
            self.visualize_result(default_file)
        else:
            self.setup_plot_style("Awaiting Hardware Data...")

    def setup_plot_style(self, title):
        self.ax.clear()
        self.ax.set_title(title, fontsize=11, fontweight='bold')
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Voltage (V)")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.fig.tight_layout()
        self.canvas.draw()

    def update_active_specs(self, rate, duration):
        """Updates the top right panel with the active target metrics"""
        total_samples = rate * duration
        self.lbl_sps.config(text=f"Sampling Rate: {rate} Hz (Target)")
        self.lbl_time.config(text=f"Recording Time: {duration} Seconds (Target)")
        self.lbl_total.config(text=f"Total Samples: {total_samples:,} (Target)")

    def log_output(self, message):
        self.root.after(0, self._append_log, message)
        
    def _append_log(self, message):
        self.txt_output.insert(tk.END, message + "\n")
        self.txt_output.see(tk.END)

    def select_db(self):
        folder = filedialog.askdirectory(title="Select ECG Database Folder")
        if folder:
            self.db_dir = folder
            self.lbl_db.config(text=f"DB Path: {os.path.basename(self.db_dir)}")
            self.log_output(f"[*] DB target updated.")

    def get_output_filepath(self):
        if self.dest_var.get() == "LIVE":
            return "live_test.txt"
        else:
            # Auto-numbering logic for Database (Odd/Even Split)
            if not os.path.exists(self.db_dir):
                os.makedirs(self.db_dir)
            import glob
            files = glob.glob(os.path.join(self.db_dir, "*.txt"))
            nums = []
            for f in files:
                try:
                    nums.append(int(os.path.basename(f).replace('.txt', '')))
                except ValueError:
                    pass
            
            # Smart determination of the next Odd or Even number
            if self.activity_state.get() == "REST":
                odds = [n for n in nums if n % 2 != 0]
                next_idx = max(odds) + 2 if odds else 1
            else: # EXERCISE
                evens = [n for n in nums if n % 2 == 0]
                next_idx = max(evens) + 2 if evens else 2
                
            return os.path.join(self.db_dir, f"{next_idx}.txt")

    def start_recording(self):
        if self.is_recording:
            return
            
        # Pull parameters safely from GUI
        try:
            # Now pulling securely from the Combobox
            rate_hz = int(self.combo_rate.get().strip())
            duration_sec = int(self.entry_duration.get().strip())
            
            if rate_hz > 860:
                messagebox.showwarning("Warning", "ADS1115 max capacity is 860 SPS. Values above 860 Hz may cause dropped data.")
                return
        except ValueError:
            messagebox.showerror("Error", "Duration must be a valid integer.")
            return

        com_port = self.combo_port.get().strip()
        output_file = self.get_output_filepath()
        
        self.is_recording = True
        self.btn_start.config(state=tk.DISABLED, text="Acquisition in Progress...")
        self.setup_plot_style(f"Recording @ {rate_hz} Hz for {duration_sec} seconds...")
        self.update_active_specs(rate_hz, duration_sec)
        
        record_thread = threading.Thread(
            target=self.serial_acquisition_task, 
            args=(com_port, output_file, rate_hz, duration_sec), 
            daemon=True
        )
        record_thread.start()

    def serial_acquisition_task(self, com_port, output_file, rate_hz, duration_sec):
        self.log_output(f"\n[*] Connecting to Arduino on {com_port}...")
        
        try:
            ser = serial.Serial(com_port, 115200, timeout=1)
            time.sleep(2) # Allow Arduino to reset upon connection
        except Exception as e:
            self.log_output(f"[-] Connection Failed: {e}")
            self.root.after(0, self.reset_ui_state)
            return

        # DYNAMIC COMMAND: Send formatted string -> S,300,300\n
        command = f"S,{rate_hz},{duration_sec}\n"
        self.log_output(f"[*] Sending Configuration: {command.strip()}")
        ser.write(command.encode('utf-8')) 

        samples_collected = 0
        total_target = rate_hz * duration_sec
        
        # Set dynamic progress update interval (e.g., report every 10% of total)
        report_interval = total_target // 10 if total_target >= 10 else 1
        
        try:
            with open(output_file, 'w') as f:
                while self.is_recording:
                    line = ser.readline().decode('utf-8').strip()
                    
                    if line == "START":
                        self.log_output("[+] Arduino acknowledged. Recording...")
                        continue
                    elif line == "END":
                        self.log_output(f"\n[+] Acquisition Complete!")
                        self.log_output(f"[*] Total samples saved: {samples_collected}")
                        break
                    
                    try:
                        float(line) 
                        f.write(line + "\n")
                        samples_collected += 1
                        
                        if samples_collected % report_interval == 0:
                            percent = (samples_collected / total_target) * 100
                            self.log_output(f" > Progress: {percent:.0f}% ({samples_collected} samples)")
                    except ValueError:
                        pass
                        
        except Exception as e:
            self.log_output(f"[-] Unexpected Error: {e}")
        finally:
            ser.close()
            self.root.after(0, self.reset_ui_state)
            if samples_collected > 100:
                self.root.after(0, lambda: self.visualize_result(output_file))

    def reset_ui_state(self):
        self.is_recording = False
        self.btn_start.config(state=tk.NORMAL, text="Start Acquisition")

    def visualize_result(self, filepath):
        try:
            raw_data = np.loadtxt(filepath)
            self.setup_plot_style(f"Hardware ECG Result: {os.path.basename(filepath)}")
            
            # Plot the first 1500 samples (to show up to 5 seconds depending on Hz)
            plot_data = raw_data[:1500] if len(raw_data) > 1500 else raw_data
            self.ax.plot(plot_data, color='#1f77b4', linewidth=1.2)
            self.canvas.draw()
            self.log_output("[+] Visualization updated.")
            
            # --- DYNAMIC CALCULATION FOR LOADED FILES ---
            total_samples = len(raw_data)
            assumed_rate = 300 # Standard project methodology rate
            calculated_duration = total_samples // assumed_rate
            
            self.lbl_sps.config(text=f"Sampling Rate: {assumed_rate} Hz (Standard)")
            self.lbl_time.config(text=f"Recording Time: ~{calculated_duration} Seconds")
            self.lbl_total.config(text=f"Total Samples Loaded: {total_samples:,}")
            # --------------------------------------------
            
        except Exception as e:
            self.log_output(f"[-] Error plotting data: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGRecorderGUI(root)
    root.mainloop()