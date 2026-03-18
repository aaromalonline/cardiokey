import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# --- 1. SIGNAL PROCESSING FUNCTIONS ---

def preprocess_ecg(data, fs=300):
    """
    Stage 1 & 2: Butterworth Bandpass (0.5Hz - 40Hz)
    and Notch Filter (60Hz). This alternative bypasses
    PyWavelets entirely while achieving the exact same
    baseline correction required by the research.
    """
    # 1. Bandpass Filter:
    # Removes Baseline Wander (< 0.5Hz)
    # and High-Frequency Noise (> 40Hz)
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 40.0 / nyquist
    b_band, a_band = signal.butter(4, [low, high],
                                   btype='bandpass')
    data_bandpassed = signal.filtfilt(b_band, a_band, data)

    # 2. Mains Hum Removal: Notch Filter exactly at 60Hz
    b_notch, a_notch = signal.iirnotch(60.0, 30.0, fs)
    data_final = signal.filtfilt(b_notch, a_notch,
                                 data_bandpassed)

    return data_final[:len(data)]


def segment_beats(signal_data, fs=300, num_beats=30):
    """
    Stage 3: Extracts P-QRS-T windows.
    Corrected for 300Hz sampling rate:
    150 samples (0.5 seconds).
    """
    threshold = np.max(signal_data) * 0.5
    peaks, _ = signal.find_peaks(signal_data,
                                 distance=int(fs * 0.6),
                                 height=threshold)

    beats = []
    for p in peaks:
        # THE ADJUSTMENT: 48 samples before R,
        # 102 after R (Total 150 samples)
        if p >= 48 and p + 102 < len(signal_data):
            beat = signal_data[p - 48: p + 102]
            # Zero-mean the fragment
            beat = beat - np.mean(beat)
            beats.append(beat)
            if len(beats) == num_beats:
                break
    return np.array(beats)


# --- 2. FEATURE EXTRACTION (PCA) & CLASSIFICATION (LDA) ---

def authenticate_users(dataset_X, dataset_Y):
    """
    Stage 4 & 5: PCA Dimensionality Reduction and LDA.
    dataset_X: Array of all 150-sample P-QRS-T fragments
    dataset_Y: Array of corresponding subject IDs
    """
    # Feature Space Reduction: 150 features -> 30 PCs
    pca = PCA(n_components=30)
    X_reduced = pca.fit_transform(dataset_X)

    # Split data: 70% Training, 30% Testing/Live Auth
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, dataset_Y, test_size=0.3, random_state=42
    )

    # Classification using Linear Discriminant Analysis
    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train, y_train)

    # Predict on the test set (Simulated Access Attempts)
    predictions = classifier.predict(X_test)

    # Performance Evaluation
    accuracy = accuracy_score(y_test, predictions) * 100
    print(f"System Authentication Accuracy: {accuracy:.2f}%")

    return classifier, pca


# --- 4. DATASET ITERATION & MAIN PIPELINE ---

if __name__ == "__main__":
    dataset_X_rest = []
    dataset_Y_rest = []

    # 45 subjects, 2 files each = 90 total files
    data_dir = 'ECGIDld2_db'

    print("Initializing Signal Processing Pipeline...")

    for i in range(1, 91):
        filepath = os.path.join(data_dir, f"{i}.txt")
        if not os.path.exists(filepath):
            continue

        # Load raw digitized data
        raw_signal = np.loadtxt(filepath)

        # Determine Subject ID and State
        # Subj 1: 1.txt, 2.txt | Subj 2: 3.txt, 4.txt
        subject_id = (i + 1) // 2
        is_exercise = (i % 2 == 0)

        # Stage 1 & 2: Clean the hardware signal
        clean_signal = preprocess_ecg(raw_signal, fs=300)

        # Stage 3: Extract 150-sample biometric fragments
        beats = segment_beats(clean_signal, fs=300, num_beats=10)

        # Append to respective datasets
        for beat in beats:
            # Filtering for resting baseline accuracy
            if not is_exercise:
                dataset_X_rest.append(beat)
                dataset_Y_rest.append(subject_id)

    # Convert lists to NumPy arrays for Scikit-Learn
    dataset_X_rest = np.array(dataset_X_rest)
    dataset_Y_rest = np.array(dataset_Y_rest)

    # Stage 4 & 5: Run Authentication Model
    if len(dataset_X_rest) > 0:
        print("Evaluating Base Resting Accuracy...")
        clf, pca = authenticate_users(dataset_X_rest,
                                      dataset_Y_rest)

        # --- 5. REAL-WORLD HARDWARE AUTHENTICATION TEST ---
        print("\n--- Real-World Hardware Access Attempt ---")
        live_file = "livetest1/live_test.txt"

        # AUTO-GENERATE SIMULATED LIVE TEST IF MISSING
        if not os.path.exists(live_file):
            print(f"'{live_file}' missing. Generating demo...")

            # Pick a random odd file (Resting data)
            rand_idx = random.choice(range(1, 91, 2))
            sim_path = os.path.join(data_dir, f"{rand_idx}.txt")
            sim_true_id = (rand_idx + 1) // 2

            print(f"Extracting unseen tail data from Subject "
                  f"{sim_true_id} to simulate hardware input.")

            # Load the 5-min recording and grab last 3000 samples
            # The first 10 beats were used, so tail is untouched!
            full_sig = np.loadtxt(sim_path)
            tail_sig = full_sig[-3000:]

            # Save to simulate Arduino serial output
            np.savetxt(live_file, tail_sig, fmt="%.6f")

        # PROCEED WITH AUTHENTICATION
        if os.path.exists(live_file):
            print(f"Loading live ECG data from: {live_file}")
            live_raw = np.loadtxt(live_file)

            # 1. Clean the hardware signal (Bandpass + Notch)
            live_clean = preprocess_ecg(live_raw, fs=300)

            # 2. Extract fragments (we need 1 valid beat)
            live_beats = segment_beats(live_clean, fs=300,
                                       num_beats=1)

            if len(live_beats) > 0:
                live_150pt = live_beats

                # 3. Compress using ALREADY TRAINED PCA
                live_30pt = pca.transform(live_150pt.reshape(1, -1))

                # 4. Predict Identity and calculate Confidence
                predicted_id = clf.predict(live_30pt)
                probabilities = clf.predict_proba(live_30pt)
                confidence = np.max(probabilities) * 100

                print(f"System Prediction: Subject {predicted_id}")
                print(f"Match Confidence: {confidence:.2f}%")

                # 5. Security Threshold Logic
                if confidence > 80.0:
                    print("Authentication Result: ACCESS GRANTED")
                else:
                    print("Auth Result: DENIED (Low Confidence)")
            else:
                print("Error: No valid heartbeat detected.")
