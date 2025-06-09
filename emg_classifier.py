import numpy as np
import pandas as pd
import os
import glob
from scipy import signal, stats
import joblib
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import antropy as ant
import warnings
import re
import time
from collections import defaultdict
warnings.filterwarnings('ignore', category=UserWarning)

GLOBAL_BASELINE_STATS = None  # Global variable to store baseline stats


def load_emg_dataset(data_dir="user_defined_emg_data_4ch", movements=['Rest', 'Flexion', 'Extension', 'Pronation', 'Supination', 'Grasp_Power'], remove_fds_channel=True):
    """
    Load all EMG data files and their corresponding labels and metadata
    """
    # Find all EMG data files
    emg_files = glob.glob(os.path.join(data_dir, "*_emgData_*.npy"))
    
    print(f"Found {len(emg_files)} EMG data files in {data_dir}")
    if len(emg_files) == 0:
        print(f"ERROR: No EMG data files found in '{data_dir}'")
        if not os.path.exists(data_dir):
            print(f"ERROR: Directory '{data_dir}' does not exist!")
    
    all_data = []
    all_labels = []
    all_trials = []
    all_subjects = []
    all_metadata = []

    fds_channel_index = 1
    
    for emg_file in emg_files:
        print(f"Processing file: {os.path.basename(emg_file)}")
        
        # Get corresponding label file
        label_file = emg_file.replace("_emgData_", "_labels_")
        label_file = label_file.replace(".npy", ".csv")
        
        # Get corresponding metadata file
        metadata_file = emg_file + ".npz"
        
        if not os.path.exists(label_file):
            print(f"Warning: No label file found for {emg_file}")
            continue
            
        # Extract subject, trial, and movement type from filename
        filename = os.path.basename(emg_file)
        
        # Improved parsing for filenames with movements that contain underscores
        # Expected format: SS01_T1_emgData_Movement_Channel1_Channel2_YYYYMMDD_HHMMSS.npy
        
        # First, reliably extract subject and trial
        match = re.match(r'([A-Za-z0-9]+)_T(\d+)_emgData_(.+)_(\w+)_(\w+)_(\d+)_(\d+)\.npy', filename)
        if match:
            subject_id = match.group(1)
            trial_num = int(match.group(2))
            
            # The movement might contain underscores and is between "_emgData_" and the first channel
            rest_of_name = match.group(3)
            
            # Determine movement by checking against the allowed movements list
            movement = None
            for allowed_movement in movements:
                if rest_of_name.startswith(allowed_movement):
                    movement = allowed_movement
                    break
            
            # If we couldn't identify the movement, try an alternative approach
            if movement is None:
                # Look for the first and second underscores after "emgData"
                emgdata_index = filename.find("_emgData_")
                if emgdata_index >= 0:
                    # Extract everything after "_emgData_"
                    remaining = filename[emgdata_index + len("_emgData_"):]
                    
                    # Try to find the movement by checking each possible movement
                    for allowed_movement in sorted(movements, key=len, reverse=True):
                        if remaining.startswith(allowed_movement):
                            movement = allowed_movement
                            break
            
            if movement is None:
                print(f"Warning: Could not extract movement from filename: {filename}")
                continue
                
            print(f"Extracted - Subject: {subject_id}, Trial: {trial_num}, Movement: {movement}")
        else:
            print(f"Warning: Filename does not match expected pattern: {filename}")
            continue
        
        if movements and movement not in movements:
            print(f"Skipping file with movement: {movement} (not in {movements})")
            continue
        
        # Load EMG data - shape should be (n_channels, n_samples)
        try:
            emg_data_original = np.load(emg_file)
            print(f"Loaded EMG data shape: {emg_data_original.shape}")
            
            # Verify shape - should be (channels, samples)
            if len(emg_data_original.shape) != 2:
                print(f"Warning: EMG data has unexpected shape: {emg_data_original.shape}")
                continue

            emg_data_processed = emg_data_original # Default to original
            channel_was_removed = False

            if remove_fds_channel:
                if emg_data_original.shape[0] > 1 and emg_data_original.shape[0] > fds_channel_index:
                    # Only attempt removal if there's more than one channel AND the index is valid
                    emg_data_processed = np.delete(emg_data_original, fds_channel_index, axis=0)
                    print(f"Removed channel at index {fds_channel_index} (FDS). New EMG data shape: {emg_data_processed.shape}")
                    channel_was_removed = True
                elif emg_data_original.shape[0] <= 1:
                    print(f"Not removing channel: Only {emg_data_original.shape[0]} channel(s) present.")
                else: # Index out of bounds for multi-channel data (e.g. trying to remove ch1 from 1-ch data)
                    print(f"Not removing channel: Index {fds_channel_index} invalid for {emg_data_original.shape[0]} channels.")
                
        except Exception as e:
            print(f"Error loading EMG data: {e}")
            continue
            
        # Load metadata if available
        metadata = {}
        if os.path.exists(metadata_file):
            try:
                metadata_npz = np.load(metadata_file)
                dc_offsets_from_file = metadata_npz['dc_offsets']
                sampling_rate_from_file = metadata_npz['sampling_rate']
                print(f"Loaded metadata from file. Original DC offsets length: {len(dc_offsets_from_file)}")

                # Adjust DC offsets from file if a channel was removed
                if channel_was_removed:
                    if len(dc_offsets_from_file) > fds_channel_index:
                        final_dc_offsets = np.delete(dc_offsets_from_file, fds_channel_index)
                        source_of_dc_offsets = "file (adjusted)"
                    else:
                        print(f"Warning: Loaded DC offsets length ({len(dc_offsets_from_file)}) "
                              f"not sufficient for removed channel index ({fds_channel_index}). Will estimate DC offsets.")
                        # final_dc_offsets remains None, will be estimated
                # If no channel removed, check if loaded DC offsets match current channel count
                elif len(dc_offsets_from_file) == emg_data_processed.shape[0]:
                     final_dc_offsets = dc_offsets_from_file
                     source_of_dc_offsets = "file"
                else: # Mismatch even if no channel was removed
                    print(f"Warning: Loaded DC offsets length ({len(dc_offsets_from_file)}) "
                          f"does not match current channel count ({emg_data_processed.shape[0]}). Will estimate DC offsets.")
                    # final_dc_offsets remains None, will be estimated

                final_sampling_rate = sampling_rate_from_file # Use sampling rate from file if loaded
                source_of_sampling_rate = "file"

            except Exception as e:
                print(f"Error processing metadata from file: {e}. Will estimate DC offsets and use default sampling rate if needed.")
                # final_dc_offsets and final_sampling_rate might still be None

        # Estimate DC offsets if not successfully determined from file
        if final_dc_offsets is None:
            final_dc_offsets = np.mean(emg_data_processed, axis=1)
            source_of_dc_offsets = "estimated" # Explicitly set if estimation occurred

        # Use default sampling rate if not successfully determined from file
        if final_sampling_rate is None:
            final_sampling_rate = 2000  # Default
            source_of_sampling_rate = "default" # Explicitly set

        metadata = {
            'dc_offsets': final_dc_offsets,
            'sampling_rate': final_sampling_rate
        }
        print(f"Using DC offsets ({source_of_dc_offsets}): {final_dc_offsets}")
        print(f"Using sampling rate ({source_of_sampling_rate}): {final_sampling_rate}")

        # Store all information
        all_data.append(emg_data_processed) # emg_data_processed has the channel removed (if applicable)
        all_labels.append(movement)
        all_trials.append(trial_num)
        all_subjects.append(subject_id)
        all_metadata.append(metadata)

        print(f"Successfully processed: Subject={subject_id}, Movement={movement}, Trial={trial_num}, Final Shape={emg_data_processed.shape}")

    print(f"Total files successfully processed: {len(all_data)}")
    return all_data, all_labels, all_trials, all_subjects, all_metadata


def preprocess_emg(emg_data, metadata, apply_filters=True):
    """
    Preprocess EMG data using metadata for offset correction
    
    Args:
        emg_data: EMG data with shape (n_channels, n_samples)
        metadata: Dictionary containing 'dc_offsets' and 'sampling_rate'
        apply_filters: Whether to apply bandpass and notch filters
        
    Returns:
        Filtered EMG data with same shape as input
    """
    n_channels, n_samples = emg_data.shape
    processed_data = np.zeros_like(emg_data)
    
    # 1. Apply DC offset correction using metadata
    dc_offsets = metadata['dc_offsets']
    sampling_rate = metadata['sampling_rate']

    if isinstance(sampling_rate, np.ndarray):
        if sampling_rate.ndim == 0:  # It's a 0-d array (scalar)
            sampling_rate = float(sampling_rate.item())
        else:  # It's a regular array
            sampling_rate = float(sampling_rate[0])
    elif isinstance(sampling_rate, (list, tuple)):
        sampling_rate = float(sampling_rate[0])
    elif not isinstance(sampling_rate, (int, float)):
        sampling_rate = float(sampling_rate)
    
    print(f"Using DC offsets from metadata: {dc_offsets}")
    
    # Design filters if needed
    if apply_filters:
        # 1. Bandpass filter design (20-450Hz)
        lowcut = 20.0  # Hz
        highcut = 450.0  # Hz
        nyquist = 0.5 * sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Check if filter parameters are valid
        if low >= 1.0 or high >= 1.0:
            print(f"Warning: Filter frequencies ({lowcut}, {highcut}) too high for sampling rate {sampling_rate}")
            low = min(0.95, low)
            high = min(0.99, high)
            
        order_bandpass = 4  # Butterworth filter order
        sos_bandpass = signal.butter(order_bandpass, [low, high], btype='band', output='sos')
        
        # 2. Notch filter design (50Hz)
        notch_freq = 50.0  # Hz (power line interference)
        quality_factor = 30.0  # Q factor - determines bandwidth
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sampling_rate)
        sos_notch = signal.tf2sos(b_notch, a_notch)
        
        print(f"Applying bandpass filter ({lowcut}-{highcut}Hz) and 50Hz notch filter")
    
    # Process each channel
    for c in range(n_channels):
        # Get channel data
        channel_data = emg_data[c, :].copy()
        
        # Apply DC offset correction
        dc_offset = dc_offsets[c] if c < len(dc_offsets) else np.mean(channel_data)
        print(f"  Channel {c}: Using DC offset = {dc_offset:.4f}")
        channel_data = channel_data - dc_offset
        
        if apply_filters:
            # Apply bandpass filter
            channel_data = signal.sosfilt(sos_bandpass, channel_data)
            
            # Apply notch filter
            channel_data = signal.sosfilt(sos_notch, channel_data)
        
        # Store processed data
        processed_data[c, :] = channel_data
    
    return processed_data


def segment_data(emg_data, window_size=500, overlap=0.75, discard_small_end=True):
    """
    Segment EMG data into windows
    
    Args:
        emg_data: EMG data with shape (n_channels, n_samples)
        window_size: Size of window in samples
        overlap: Overlap between consecutive windows (0-1)
        discard_small_end: Whether to discard the last window if it's smaller than window_size
    
    Returns:
        Segmented data with shape (n_windows, n_channels, window_size)
    """
    n_channels, n_samples = emg_data.shape
    stride = int(window_size * (1 - overlap))
    
    # Calculate number of windows
    n_windows = (n_samples - window_size) // stride + 1
    if n_windows <= 0:
        print(f"Warning: Not enough samples ({n_samples}) for window size {window_size}")
        return np.empty((0, n_channels, window_size))
    
    # Initialize segmented data
    segmented = np.zeros((n_windows, n_channels, window_size))
    
    # Segment data
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        
        # Check if we have enough samples
        if end > n_samples:
            if discard_small_end:
                # Reduce number of windows
                segmented = segmented[:i]
                break
            else:
                # Pad with zeros
                segment = np.zeros((n_channels, window_size))
                segment[:, :n_samples-start] = emg_data[:, start:n_samples]
                segmented[i] = segment
        else:
            segmented[i] = emg_data[:, start:end]
    
    return segmented


def detect_activity_regions_single_movement_dt(processed_data, metadata, threshold_multiplier=2, min_duration_ms=100, apply_longest_run_logic=True, global_baseline_stats=None, use_adaptive_baseline=False, pre_rest_duration=1.0): 
    # Calculate signal energy (combined across channels)
    energy = np.mean(np.square(processed_data), axis=0)
    sampling_rate = metadata['sampling_rate']

    # Smooth the energy
    smoothing_window_ms = 100
    smoothing_window_samples = int(smoothing_window_ms / 1000 * sampling_rate)
    if smoothing_window_samples < 1: smoothing_window_samples = 1
    if smoothing_window_samples % 2 == 0: smoothing_window_samples += 1
    window = np.ones(smoothing_window_samples) / smoothing_window_samples
    smoothed_energy = np.convolve(energy, window, mode='same')

    # CHANGE: Add adaptive baseline logic here
    if use_adaptive_baseline:
        pre_rest_samples = int(pre_rest_duration * sampling_rate)
        rest_baseline_energy = smoothed_energy[:pre_rest_samples]
        # Use 95th percentile of rest energy instead of mean + std
        amplitude_threshold = np.percentile(rest_baseline_energy, 95) * threshold_multiplier
        print(f"Percentile-based threshold: {amplitude_threshold:.4f}")
    else:
        # Original logic
        if global_baseline_stats is not None:
            amplitude_threshold = global_baseline_stats["mean"] + \
                                  threshold_multiplier * global_baseline_stats["std"]
            print(f"Activity Detection: Using global baseline. Amplitude Threshold={amplitude_threshold:.4f}")
        else:
            baseline = np.percentile(smoothed_energy, 10)
            amplitude_threshold = baseline * threshold_multiplier
            print(f"Activity Detection: Using per-trial baseline. Baseline Energy={baseline:.4f}, Amplitude Threshold={amplitude_threshold:.4f}")

    # Rest of function remains exactly the same...


    min_duration_samples = int(min_duration_ms / 1000 * sampling_rate)
    if min_duration_samples < 1: min_duration_samples = 1
    print(f"Activity Detection: Min Duration Samples={min_duration_samples}")

    # Find regions where smoothed_energy > amplitude_threshold
    potential_active_regions = smoothed_energy > amplitude_threshold
    # print(f"Samples above amplitude threshold: {np.sum(potential_active_regions)}")

    # Apply duration threshold
    valid_runs = []
    in_run = False
    run_start_index = 0
    for i, active in enumerate(potential_active_regions):
        if active and not in_run: # Start of a potential run
            in_run = True
            run_start_index = i
        elif not active and in_run: # End of a potential run
            in_run = False
            run_length = i - run_start_index
            if run_length >= min_duration_samples:
                valid_runs.append((run_start_index, run_length))
    if in_run: # Handle case where activity extends to the end
        run_length = len(potential_active_regions) - run_start_index
        if run_length >= min_duration_samples:
            valid_runs.append((run_start_index, run_length))

    final_activity_mask = np.zeros_like(potential_active_regions, dtype=bool)
    if not valid_runs:
        print("No valid activity runs found after applying duration threshold.")
        return final_activity_mask, smoothed_energy

    if apply_longest_run_logic and valid_runs: # Ensure valid_runs is not empty
        if len(valid_runs) == 1:
            best_run_start, best_run_length = valid_runs[0]
        else:
            sorted_runs_by_length = sorted(valid_runs, key=lambda x: x[1], reverse=True)
            if sorted_runs_by_length[0][1] > (sorted_runs_by_length[1][1] * 1.5 if len(sorted_runs_by_length) > 1 else 0): # check if more than one run
                best_run_start, best_run_length = sorted_runs_by_length[0]
            else:
                run_energies = []
                for start_idx, length in sorted_runs_by_length[:min(3, len(sorted_runs_by_length))]:
                    avg_energy_in_run = np.mean(smoothed_energy[start_idx : start_idx + length])
                    run_energies.append((start_idx, length, avg_energy_in_run))
                if run_energies: # Ensure run_energies is not empty
                     best_run_start, best_run_length, _ = max(run_energies, key=lambda x: x[2])
                else: # Fallback if no runs qualified for energy comparison (e.g. all runs were too short)
                    if valid_runs: # Should always be true if we reached here and apply_longest_run_logic is true
                         best_run_start, best_run_length = valid_runs[0] # Default to the first valid run
                    else: # Should not happen if logic is correct
                        return final_activity_mask, smoothed_energy


        final_activity_mask[best_run_start : best_run_start + best_run_length] = True
        print(f"Selected best run: start={best_run_start}, length={best_run_length}")
    elif valid_runs: # Not applying longest run logic, or it wasn't applicable
        for start_idx, length in valid_runs:
            final_activity_mask[start_idx : start_idx + length] = True
        print(f"Marked {len(valid_runs)} valid runs in the activity mask.")
    # If no valid_runs and not apply_longest_run_logic, mask remains all False.

    return final_activity_mask, smoothed_energy

def label_windows_with_activity(segmented_data, activity_mask, window_size, overlap, movement_label):
    """
    Label windows based on activity detection.

    Returns:
        window_labels: Array of labels for each window.
        window_center_samples: List of sample indices for the center of each window.
    """
    n_windows = segmented_data.shape[0]
    stride = int(window_size * (1 - overlap))
    window_labels = []
    window_center_samples = [] # NEW: To store center positions

    for i in range(n_windows):
        start_pos = i * stride
        center_pos = start_pos + window_size // 2
        window_center_samples.append(center_pos) # Store center position

        if center_pos < len(activity_mask):
            is_active = activity_mask[center_pos]
            if movement_label == "Rest":
                window_labels.append("Rest")
            elif is_active:
                window_labels.append(movement_label)
            else:
                window_labels.append("Transition")
        else:
            window_labels.append("Unknown")

    return window_labels, window_center_samples # Return both


def apply_bandpass_filter(signal, low_freq, high_freq, sampling_rate):
    """Apply bandpass filter to signal"""
    from scipy import signal as sig
    nyquist = 0.5 * sampling_rate
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Ensure filter parameters are valid
    if low >= 1.0 or high >= 1.0:
        low = min(0.95, low)
        high = min(0.99, high)
    
    if low >= high:
        return signal  # Return original if invalid range
        
    try:
        sos = sig.butter(4, [low, high], btype='band', output='sos')
        filtered = sig.sosfilt(sos, signal)
        return filtered
    except:
        return signal  # Return original if filter fails


def apply_bandpass_filter(signal, low_freq, high_freq, sampling_rate):
    try:
        from scipy import signal as sig
        nyquist = 0.5 * sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure filter parameters are valid
        if low >= 1.0 or high >= 1.0:
            low = min(0.95, low)
            high = min(0.99, high)
        
        if low >= high:
            return signal  # Return original if invalid range
            
        sos = sig.butter(4, [low, high], btype='band', output='sos')
        filtered = sig.sosfilt(sos, signal)
        return filtered
    except Exception as e:
        print(f"Bandpass filter failed: {e}")
        return signal  # Return original if filter fails

def enhanced_rms_features(signal, sampling_rate):
    """Extract RMS in different frequency bands"""
    try:
        # Low band (20-60 Hz) - slow motor unit activity
        low_filtered = apply_bandpass_filter(signal, 20, 60, sampling_rate)
        rms_low = np.sqrt(np.mean(low_filtered**2))
        
        # Mid band (60-150 Hz) - main EMG content
        mid_filtered = apply_bandpass_filter(signal, 60, 150, sampling_rate)
        rms_mid = np.sqrt(np.mean(mid_filtered**2))
        
        # High band (150-450 Hz) - fast motor unit activity
        high_filtered = apply_bandpass_filter(signal, 150, 450, sampling_rate)
        rms_high = np.sqrt(np.mean(high_filtered**2))
        
        return [rms_low, rms_mid, rms_high]
    except Exception as e:
        print(f"Enhanced RMS features failed: {e}")
        return [0.0, 0.0, 0.0]

def cross_channel_features(window):
    n_channels = window.shape[0]
    correlations = []
    
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            try:
                corr = np.corrcoef(window[i], window[j])[0,1]
                correlations.append(corr if not np.isnan(corr) else 0)
            except:
                correlations.append(0)
    
    return correlations

def extract_features_with_context(windows, history_size=3, sampling_rate=2000):
    """
    Extract enhanced features from segmented EMG data with temporal context - UPDATED WITH WELCH
    """
    if windows.shape[0] == 0:
        return np.array([])

    n_windows, n_data_channels, window_size = windows.shape
    epsilon = 1e-10

    # Define enhanced features per channel - UPDATED COUNTING
    num_basic_features_per_channel = 12  # Original features (including spectral)
    num_enhanced_rms_features = 3        # RMS in 3 frequency bands
    
    # Total features per channel
    num_features_per_channel = num_basic_features_per_channel + num_enhanced_rms_features
    
    print(f"Features per channel: {num_basic_features_per_channel} basic + {num_enhanced_rms_features} RMS = {num_features_per_channel}")
    
    # Cross-channel correlation features
    num_cross_channel_features = (n_data_channels * (n_data_channels - 1)) // 2
    
    # Ratio features
    num_ratio_features = 0
    has_bt_ratio = False
    has_fe_ratio = False

    # Determine channel indices and ratio features
    idx_biceps = 0
    idx_triceps = -1
    idx_fds = -1
    idx_edc = -1

    if n_data_channels >= 2:
        idx_triceps = 1 if n_data_channels == 2 else (2 if n_data_channels >= 3 else -1)
        if idx_biceps < n_data_channels and idx_triceps != -1 and idx_triceps < n_data_channels:
             num_ratio_features += 1
             has_bt_ratio = True

    if n_data_channels == 4:
        idx_fds = 1
        idx_edc = 3
        if idx_fds < n_data_channels and idx_edc < n_data_channels:
            num_ratio_features += 1
            has_fe_ratio = True

    # Total features for current window
    num_current_window_features = (n_data_channels * num_features_per_channel) + num_cross_channel_features + num_ratio_features
    
    print(f"Total features per window: {n_data_channels} channels * {num_features_per_channel} + {num_cross_channel_features} cross-channel + {num_ratio_features} ratios = {num_current_window_features}")
    
    # Array to store features
    features_current_window_array = np.zeros((n_windows, num_current_window_features))

    for w in range(n_windows):
        # Store MAVs for ratio calculation
        mavs_for_this_window = np.zeros(n_data_channels)
        
        feature_idx = 0  # Track current feature index

        # Extract features for each channel
        for c in range(n_data_channels):
            signal_data = windows[w, c]
            
            # Original basic features
            mav = np.mean(np.abs(signal_data))
            mavs_for_this_window[c] = mav
            
            rms = np.sqrt(np.mean(signal_data**2))
            current_std = np.std(signal_data)
            zc_threshold = 0.015 * current_std if current_std > 0 else 0
            zc = np.sum(np.diff(np.signbit(signal_data)) & (np.abs(np.diff(signal_data)) > zc_threshold))
            wl = np.sum(np.abs(np.diff(signal_data)))
            
            try:
                import antropy as ant
                entropy = ant.sample_entropy(signal_data) if window_size > 1 and len(np.unique(signal_data)) > 1 else 0
            except:
                entropy = 0
                
            variance = np.var(signal_data)
            
            # UPDATED: Use Welch's method for spectral features
            mean_freq, median_freq, peak_freq, low_band_power, med_band_power, high_band_power = extract_spectral_features_welch(
                signal_data, sampling_rate, window_size
            )
            
            # Enhanced RMS features
            enhanced_rms = enhanced_rms_features(signal_data, sampling_rate)
            
            # Combine all channel features
            basic_features = [
                mav, rms, zc, wl, entropy, variance,
                mean_freq, median_freq, peak_freq, 
                low_band_power, med_band_power, high_band_power
            ]
            
            channel_features = basic_features + enhanced_rms
            
            # Verify total channel features length
            if len(channel_features) != num_features_per_channel:
                print(f"Warning: Channel {c} features mismatch. Expected {num_features_per_channel}, got {len(channel_features)}")
                # Pad or truncate to match expected length
                if len(channel_features) < num_features_per_channel:
                    channel_features.extend([0.0] * (num_features_per_channel - len(channel_features)))
                else:
                    channel_features = channel_features[:num_features_per_channel]
            
            # Store channel features
            features_current_window_array[w, feature_idx:feature_idx + num_features_per_channel] = channel_features
            feature_idx += num_features_per_channel
        
        # Cross-channel correlation features
        cross_corr_feats = cross_channel_features(windows[w])
        if len(cross_corr_feats) != num_cross_channel_features:
            print(f"Warning: Cross-channel features mismatch. Expected {num_cross_channel_features}, got {len(cross_corr_feats)}")
            if len(cross_corr_feats) < num_cross_channel_features:
                cross_corr_feats.extend([0.0] * (num_cross_channel_features - len(cross_corr_feats)))
            else:
                cross_corr_feats = cross_corr_feats[:num_cross_channel_features]
        
        features_current_window_array[w, feature_idx:feature_idx + num_cross_channel_features] = cross_corr_feats
        feature_idx += num_cross_channel_features
        
        # Ratio features
        if has_bt_ratio:
            mav_b = mavs_for_this_window[idx_biceps]
            mav_t = mavs_for_this_window[idx_triceps]
            antagonistic_ratio_bt = mav_b / (mav_t + epsilon) if mav_t + epsilon != 0 else (mav_b / epsilon if mav_b != 0 else 0)
            features_current_window_array[w, feature_idx] = antagonistic_ratio_bt
            feature_idx += 1

        if has_fe_ratio:
            mav_f = mavs_for_this_window[idx_fds] 
            mav_e = mavs_for_this_window[idx_edc]
            antagonistic_ratio_fe = mav_f / (mav_e + epsilon) if mav_e + epsilon != 0 else (mav_f / epsilon if mav_f != 0 else 0)
            features_current_window_array[w, feature_idx] = antagonistic_ratio_fe
            feature_idx += 1

    print(f"Extracted features shape: {features_current_window_array.shape}")

    # Add temporal context (delta features)
    if history_size <= 0 or n_windows <= history_size:
        return features_current_window_array

    context_features = np.zeros((n_windows, num_current_window_features * 2))
    context_features[:history_size, :num_current_window_features] = features_current_window_array[:history_size]

    for w_idx in range(history_size, n_windows):
        context_features[w_idx, :num_current_window_features] = features_current_window_array[w_idx]
        history_avg = np.mean(features_current_window_array[w_idx-history_size:w_idx], axis=0)
        context_features[w_idx, num_current_window_features:] = features_current_window_array[w_idx] - history_avg
    
    print(f"Final features with context shape: {context_features.shape}")

    return context_features
def prepare_emg_data_with_context(data, label, metadata,
                                  trial_info_for_plot="UnknownTrial",
                                  window_size=500, overlap=0.75, history_size=3,
                                  global_baseline_stats=None,
                                  plot_each_trial=False,
                                  use_adaptive_baseline=True,
                                  pre_rest_duration=1.0,
                                  post_rest_duration=1.0):
    """
    Prepare EMG data with activity detection and temporal context - FIXED VERSION
    """
    processed_data = preprocess_emg(data, metadata)
    segmented = segment_data(processed_data, window_size, overlap)

    window_final_labels = []
    window_centers_for_plot = []
    smoothed_energy_for_plot = np.array([])
    activity_mask_for_plot = np.array([])
    amplitude_threshold_for_plot = 0

    if segmented.shape[0] == 0:
        print(f"Warning: No windows segmented for trial {trial_info_for_plot}, skipping feature extraction and plotting.")
        return np.array([]), []

    # FIXED: Define movement-specific parameters
    subtle_movements = ["Pronation", "Supination", "Flexion", "Extension", "Rest", "Grasp_Power"]
    power_movements = ["Grasp_Power"]
    
    # Get movement-specific activity detection parameters
    threshold_multiplier, min_duration_ms = get_activity_detection_params(label)

    if label == "Rest":
        # For rest, all windows are labeled "Rest"
        window_final_labels = ["Rest"] * segmented.shape[0]
        stride = int(window_size * (1 - overlap))
        for i in range(segmented.shape[0]):
            window_centers_for_plot.append(i * stride + window_size // 2)

        if plot_each_trial:
            # Calculate energy for plotting
            energy_calc = np.mean(np.square(processed_data), axis=0)
            sampling_rate_calc = metadata['sampling_rate']
            smoothed_energy_for_plot = calculate_smoothed_energy(energy_calc, sampling_rate_calc)
            activity_mask_for_plot = np.zeros_like(smoothed_energy_for_plot, dtype=bool)
            if global_baseline_stats:
                amplitude_threshold_for_plot = global_baseline_stats["mean"] + \
                                              threshold_multiplier * global_baseline_stats["std"]
            else:
                amplitude_threshold_for_plot = np.percentile(smoothed_energy_for_plot, 10) * 1.5

    elif label in subtle_movements:
        print(f"Processing subtle movement '{label}' - skipping activity detection entirely")
        
        # CHANGE: Add adaptive baseline option for realistic trials
        if use_adaptive_baseline:
            # Create realistic window labels for subtle movements too
            print(f"Creating realistic window labels for subtle movement '{label}'")
            window_final_labels = create_realistic_window_labels_for_trial(
                data, label, metadata, window_size, overlap,
                pre_rest_duration, post_rest_duration
            )
        else:
            # Original logic
            print(f"Subtle movement '{label}' - using fixed window labels")
            window_final_labels = [label] * segmented.shape[0]
        
        stride = int(window_size * (1 - overlap))
        window_centers_for_plot = [i * stride + window_size // 2 for i in range(segmented.shape[0])]

        # Print results for subtle movements
        print(f"  Total windows: {len(window_final_labels)}")
    
        # Count actual labels
        label_counts = {}
        for lbl in window_final_labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
        # Print actual distribution
        for lbl, count in label_counts.items():
            print(f"  {lbl}: {count} windows ({count/len(window_final_labels)*100:.1f}%)")

    else:  # Regular movements (Flexion, Extension, Grasp_Power, etc.)
        # CHANGE: Pass adaptive baseline parameters
        activity_mask, smoothed_energy = detect_activity_regions_single_movement_dt(
            processed_data,
            metadata,
            threshold_multiplier=threshold_multiplier,
            min_duration_ms=min_duration_ms,
            apply_longest_run_logic=True,
            global_baseline_stats=global_baseline_stats,
            use_adaptive_baseline=use_adaptive_baseline,  # NEW
            pre_rest_duration=pre_rest_duration           # NEW
        )
        
        # CHANGE: Add realistic window labeling option
        if use_adaptive_baseline:
            # Create base realistic labels
            base_window_labels = create_realistic_window_labels_for_trial(
                data, label, metadata, window_size, overlap,
                pre_rest_duration, post_rest_duration
            )
            
            # Refine with activity detection
            window_final_labels = []
            stride = int(window_size * (1 - overlap))
            
            for i, base_label in enumerate(base_window_labels):
                window_start = i * stride
                window_center = window_start + window_size // 2
                
                if base_label == label:  # Movement windows
                    if window_center < len(activity_mask):
                        window_end = min(window_start + window_size, len(activity_mask))
                        window_activity = activity_mask[window_start:window_end]
                        activity_ratio = np.sum(window_activity) / len(window_activity) if len(window_activity) > 0 else 0
                        
                        if activity_ratio >= 0.3:  # 30% of window must be active
                            window_final_labels.append(label)
                        else:
                            window_final_labels.append("Transition")
                    else:
                        window_final_labels.append("Transition")
                else:  # Rest windows
                    window_final_labels.append(base_label)
            
            # Calculate window centers for plotting
            window_centers_for_plot = [i * stride + window_size // 2 for i in range(len(window_final_labels))]
            
        else:
            # Original logic
            window_final_labels, window_centers_for_plot = label_windows_with_activity_improved(
                segmented, activity_mask, window_size, overlap, label
            )

        # Print activity detection results
        label_counts = {}
        for lbl in window_final_labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        if window_final_labels:
            print(f"Activity detection results for {trial_info_for_plot}:")
            print(f"  Total windows: {len(window_final_labels)}")
            for lbl, count in label_counts.items():
                print(f"  {lbl}: {count} windows ({count/len(window_final_labels)*100:.1f}%)")

    # Plotting (if enabled)
    if plot_each_trial and smoothed_energy_for_plot.size > 0:
        os.makedirs("activities", exist_ok=True)
        sampling_rate = metadata['sampling_rate']
        num_samples_energy = len(smoothed_energy_for_plot)
        time_vector_energy = np.linspace(0, num_samples_energy / sampling_rate, num_samples_energy)
        
        plot_filename = f"activity_plot_{trial_info_for_plot}_{label}.png"
        plot_filename = plot_filename.replace("/", "_").replace("\\", "_")
        plot_filename = "activities/" + plot_filename 

        plot_activity_detection_and_window_labels(
            time_vector_energy,
            smoothed_energy_for_plot,
            amplitude_threshold_for_plot,
            activity_mask_for_plot,
            window_centers_for_plot,
            window_final_labels,
            label,
            sampling_rate,
            output_filename=plot_filename
        )

    # Extract features
    features = extract_features_with_context(segmented, history_size)
    return features, window_final_labels

def get_activity_detection_params(movement_label):
    """
    Get movement-specific activity detection parameters
    """
    params = {
        "Rest": (2.0, 100),  # threshold_multiplier, min_duration_ms
        "Flexion": (3.0, 100),
        "Extension": (3.0, 100),
        "Pronation": (2.0, 80),      # Lower threshold, shorter duration for subtle movement
        "Supination": (2.0, 80),     # Lower threshold, shorter duration for subtle movement
        "Grasp_Power": (3.5, 120)    # Higher threshold, longer duration for power movement
    }
    
    return params.get(movement_label, (3.0, 100))  # Default values

def calculate_smoothed_energy(energy, sampling_rate, smoothing_window_ms=100):
    """
    Calculate smoothed energy signal
    """
    smoothing_window_samples = int(smoothing_window_ms / 1000 * sampling_rate)
    if smoothing_window_samples < 1: 
        smoothing_window_samples = 1
    if smoothing_window_samples % 2 == 0: 
        smoothing_window_samples += 1
    window = np.ones(smoothing_window_samples) / smoothing_window_samples
    return np.convolve(energy, window, mode='same')

def label_windows_with_activity_improved(segmented_data, activity_mask, window_size, overlap, movement_label):
    """
    Improved window labeling that considers activity over the entire window, not just center
    """
    n_windows = segmented_data.shape[0]
    stride = int(window_size * (1 - overlap))
    window_labels = []
    window_center_samples = []

    for i in range(n_windows):
        start_pos = i * stride
        end_pos = start_pos + window_size
        center_pos = start_pos + window_size // 2
        window_center_samples.append(center_pos)

        if end_pos <= len(activity_mask):
            # FIXED: Check activity over the entire window, not just center
            window_activity = activity_mask[start_pos:end_pos]
            activity_ratio = np.sum(window_activity) / len(window_activity)
            
            # Use a threshold for activity ratio (e.g., 30% of window must be active)
            activity_threshold = 0.3
            is_active = activity_ratio >= activity_threshold
            
            if movement_label == "Rest":
                window_labels.append("Rest")
            elif is_active:
                window_labels.append(movement_label)
            else:
                window_labels.append("Transition")
        else:
            window_labels.append("Unknown")

    return window_labels, window_center_samples

def identify_movement_events_with_duration(window_labels, window_times=None, confidences=None,
                                          min_duration_seconds=0.5, confidence_threshold=0.6,
                                          window_stride_ms=50):
    """
    Identify movement events with duration and confidence thresholds - FIXED VERSION
    """
    # Add type validation and fix parameter handling
    try:
        min_duration_seconds = float(min_duration_seconds)
        confidence_threshold = float(confidence_threshold)
        window_stride_ms = float(window_stride_ms)
    except (TypeError, ValueError) as e:
        print(f"Error in identify_movement_events_with_duration parameters:")
        print(f"min_duration_seconds = {min_duration_seconds} (type: {type(min_duration_seconds)})")
        print(f"confidence_threshold = {confidence_threshold} (type: {type(confidence_threshold)})")
        print(f"window_stride_ms = {window_stride_ms} (type: {type(window_stride_ms)})")
        raise
    
    n_windows = len(window_labels)
    
    if window_times is None:
        window_times = [i * window_stride_ms / 1000.0 for i in range(n_windows)]
    
    if confidences is None:
        confidences = [1.0] * len(window_labels)
        
    # Convert minimum duration to window count
    min_windows = max(1, int(min_duration_seconds * 1000 / window_stride_ms))
    
    events = []
    current_movement = "Rest"
    movement_start_idx = 0
    
    if n_windows > 0:
        movement_start_time = window_times[0]
    else:
        movement_start_time = 0
        
    confident_windows = 0
    
    # Define all possible movements (not just Extension/Flexion)
    valid_movements = ["Flexion", "Extension", "Pronation", "Supination", "Grasp_Power"]
    
    for i, (label, time, conf) in enumerate(zip(window_labels, window_times, confidences)):
        # Handle Transition/Unknown labels more robustly
        if label in ["Transition", "Unknown"]:
            # If currently tracking a movement, continue tracking it
            if current_movement != "Rest":
                continue
            else:
                label = "Rest"
        
        # Check if movement type changed
        if label != current_movement:
            # If we were tracking a non-Rest movement
            if current_movement != "Rest" and current_movement in valid_movements:
                # Calculate duration of the completed movement
                duration = time - movement_start_time
                
                # FIXED: Use movement-specific thresholds
                movement_min_duration = get_movement_specific_min_duration(current_movement, min_duration_seconds)
                movement_min_windows = max(1, int(movement_min_duration * 1000 / window_stride_ms))
                
                # Only count as event if it meets minimum duration and had enough confident windows
                if duration >= movement_min_duration and confident_windows >= movement_min_windows:
                    events.append({
                        'movement': current_movement,
                        'start_idx': movement_start_idx,
                        'end_idx': i - 1,
                        'start_time': movement_start_time,
                        'end_time': window_times[i-1] if i > 0 else time,
                        'duration': duration,
                        'confidence': sum(confidences[movement_start_idx:i]) / max(1, (i - movement_start_idx)) if i > movement_start_idx else 0
                    })
                    #print(f"Detected {current_movement} event: {duration:.2f}s duration, {confident_windows} confident windows")
            
            # Start tracking new movement if it's a valid movement (not Rest)
            if label != "Rest" and label in valid_movements:
                movement_start_idx = i
                movement_start_time = time
                confident_windows = 1 if conf >= confidence_threshold else 0
            else:
                confident_windows = 0
                
            current_movement = label
        elif label != "Rest" and label in valid_movements and conf >= confidence_threshold:
            # Same movement, increase confident window count
            confident_windows += 1
    
    # Handle case where recording ends during a movement
    if current_movement != "Rest" and current_movement in valid_movements and n_windows > 0:
        duration = window_times[-1] - movement_start_time
        movement_min_duration = get_movement_specific_min_duration(current_movement, min_duration_seconds)
        movement_min_windows = max(1, int(movement_min_duration * 1000 / window_stride_ms))
        
        if duration >= movement_min_duration and confident_windows >= movement_min_windows:
            events.append({
                'movement': current_movement,
                'start_idx': movement_start_idx,
                'end_idx': len(window_labels) - 1,
                'start_time': movement_start_time,
                'end_time': window_times[-1],
                'duration': duration,
                'confidence': sum(confidences[movement_start_idx:]) / max(1, (len(window_labels) - movement_start_idx)) if len(window_labels) > movement_start_idx else 0
            })
            print(f"Detected {current_movement} event at end: {duration:.2f}s duration, {confident_windows} confident windows")
    
    return events

def get_movement_specific_min_duration(movement, default_duration):
    """
    Get movement-specific minimum duration thresholds
    """
    # Define movement-specific thresholds
    movement_thresholds = {
        "Flexion": default_duration,
        "Extension": default_duration,
        "Pronation": default_duration * 0.7,  # Slightly shorter for subtle movements
        "Supination": default_duration * 0.7,  # Slightly shorter for subtle movements
        "Grasp_Power": default_duration * 1.2,  # Longer for power movements
    }
    
    return movement_thresholds.get(movement, default_duration)


def calculate_realistic_movement_metrics(true_labels, pred_labels, confidences=None, window_timestamps=None, sampling_rate=2000, window_size=500, overlap=0.75, min_duration_seconds=0.5, confidence_threshold=0.6):
    # Convert inputs to lists to avoid NumPy boolean evaluation issues
    true_labels = list(true_labels) if true_labels is not None else []
    pred_labels = list(pred_labels) if pred_labels is not None else []
    n_windows = len(true_labels)
    
    if n_windows == 0:
        print("No windows to analyze")
        return {
            'true_movement_count': 0,
            'detected_movement_count': 0,
            'detection_rate': 0,
            'false_activation_rate': 0,
            'detection_latencies': [],
            'mean_detection_latency': 0,
            'per_movement_metrics': {}
        }
    
    # Calculate window stride in milliseconds
    stride = int(window_size * (1 - overlap))
    window_stride_ms = stride / sampling_rate * 1000
    
    if window_timestamps is None:
        # Create timestamps based on window stride
        window_times = [i * window_stride_ms / 1000.0 for i in range(n_windows)]
    else:
        # Convert to list to avoid NumPy boolean evaluation issues
        window_times = [float(t) for t in window_timestamps]  # Ensure float format
    
    # Set default confidences if not provided
    if confidences is None:
        confidences = [1.0] * n_windows
    else:
        # Convert to list to avoid NumPy boolean evaluation issues
        confidences = [float(c) for c in confidences]
    
    # Calculate total recording time in minutes
    total_time_minutes = (window_times[-1] - window_times[0]) / 60.0 if n_windows > 0 else 0
    
    print(f"Analyzing {n_windows} windows over {total_time_minutes:.2f} minutes")
    print(f"Window stride: {window_stride_ms:.1f}ms, Min duration: {min_duration_seconds}s")
    
    # Identify true movement events using ground truth labels
    true_events = identify_movement_events_with_duration(
        true_labels, window_times, 
        min_duration_seconds=min_duration_seconds, 
        window_stride_ms=window_stride_ms
    )
    print(f"Identified {len(true_events)} true movement events")
    
    # Identify predicted movement events
    pred_events = identify_movement_events_with_duration(
        pred_labels, window_times, confidences,
        min_duration_seconds=min_duration_seconds, 
        confidence_threshold=confidence_threshold,
        window_stride_ms=window_stride_ms
    )
    print(f"Identified {len(pred_events)} predicted movement events")
    
    # Initialize metrics
    metrics = {
        'true_movement_count': len(true_events),
        'detected_movement_count': len(pred_events),
        'detection_rate': 0,
        'false_activation_rate': 0,
        'detection_latencies': [],
        'mean_detection_latency': 0,
        'per_movement_metrics': defaultdict(dict)
    }
    
    if len(true_events) == 0:
        print("No true movement events found - check min_duration_seconds parameter")
        return metrics
    
    # Count movements by type
    movement_types = {}
    for event in true_events:
        movement = event['movement']
        if movement not in movement_types:
            movement_types[movement] = 0
        movement_types[movement] += 1
    
    print(f"True movement distribution: {movement_types}")
    
    # Initialize per-movement metrics
    for movement in movement_types:
        metrics['per_movement_metrics'][movement] = {
            'true_count': movement_types[movement],
            'detected_count': 0,
            'detection_rate': 0,
            'latencies': []
        }
    
    # Match predicted events to true events with improved matching logic
    matched_true_events = [False] * len(true_events)
    matched_pred_events = [False] * len(pred_events)
    
    # For each true event, find the best matching predicted event
    for i, true_event in enumerate(true_events):
        true_movement = true_event['movement']
        true_start = true_event['start_time']
        true_end = true_event['end_time']
        true_center = (true_start + true_end) / 2
        
        best_match_idx = -1
        best_overlap = 0
        best_latency = float('inf')
        
        # Look for best matching predicted event
        for j, pred_event in enumerate(pred_events):
            if matched_pred_events[j]:
                continue  # Already matched
                
            pred_movement = pred_event['movement']
            pred_start = pred_event['start_time']
            pred_end = pred_event['end_time']
            pred_center = (pred_start + pred_end) / 2
            
            # Only consider events of the same movement type
            if pred_movement != true_movement:
                continue
            
            # Calculate temporal overlap
            overlap_start = max(true_start, pred_start)
            overlap_end = min(true_end, pred_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Calculate overlap ratio (relative to true event duration)
            true_duration = true_end - true_start
            overlap_ratio = overlap_duration / true_duration if true_duration > 0 else 0
            
            # Require minimum 30% overlap
            if overlap_ratio >= 0.3:
                # Calculate latency (distance between centers)
                latency = abs(pred_center - true_center)
                
                # Prefer higher overlap, then lower latency
                if overlap_ratio > best_overlap or (overlap_ratio == best_overlap and latency < best_latency):
                    best_match_idx = j
                    best_overlap = overlap_ratio
                    best_latency = (pred_start - true_start) * 1000  # Convert to ms
        
        # If we found a good match, record it
        if best_match_idx >= 0:
            matched_true_events[i] = True
            matched_pred_events[best_match_idx] = True
            
            metrics['detection_latencies'].append(best_latency)
            metrics['per_movement_metrics'][true_movement]['latencies'].append(best_latency)
            metrics['per_movement_metrics'][true_movement]['detected_count'] += 1
            
            print(f"Matched {true_movement}: overlap={best_overlap:.2f}, latency={best_latency:.1f}ms")
    
    # Calculate overall detection rate
    matched_true_count = sum(matched_true_events)
    metrics['detection_rate'] = matched_true_count / len(true_events)
    
    # Calculate false activations (predicted events that don't match any true event)
    false_activations = len(pred_events) - sum(matched_pred_events)
    metrics['false_activation_rate'] = false_activations / total_time_minutes if total_time_minutes > 0 else 0
    
    # Calculate latency statistics
    if metrics['detection_latencies']:
        metrics['mean_detection_latency'] = np.mean(metrics['detection_latencies'])
        metrics['median_detection_latency'] = np.median(metrics['detection_latencies'])
        metrics['std_detection_latency'] = np.std(metrics['detection_latencies'])
    
    # Calculate per-movement detection rates
    for movement, movement_metrics in metrics['per_movement_metrics'].items():
        if movement_metrics['true_count'] > 0:
            movement_metrics['detection_rate'] = (movement_metrics['detected_count'] / 
                                                 movement_metrics['true_count'])
        if movement_metrics['latencies']:
            movement_metrics['mean_latency'] = np.mean(movement_metrics['latencies'])
            movement_metrics['median_latency'] = np.median(movement_metrics['latencies'])
            movement_metrics['std_latency'] = np.std(movement_metrics['latencies'])
    
    # Print summary
    print(f"\nMovement Detection Summary:")
    print(f"  Overall Detection Rate: {metrics['detection_rate']:.2%}")
    print(f"  False Activations: {false_activations} ({metrics['false_activation_rate']:.2f}/min)")
    if metrics['detection_latencies']:
        print(f"  Mean Latency: {metrics['mean_detection_latency']:.1f}ms")
    
    return metrics

def extract_spectral_features_welch(signal_data, sampling_rate, window_size):
    """
    Extract spectral features using Welch's method for more robust PSD estimation
    """
    try:
        from scipy import signal
        
        # Use Welch's method for power spectral density
        # For 500 samples, use smaller segments (e.g., 128 samples with 50% overlap)
        nperseg = min(128, window_size // 2)  # Segment length
        noverlap = nperseg // 2  # 50% overlap
        
        freqs, psd = signal.welch(signal_data, fs=sampling_rate, 
                                 nperseg=nperseg, 
                                 noverlap=noverlap,
                                 window='hann')
        
        # Remove DC component
        freqs = freqs[1:]
        psd = psd[1:]
        
        total_power = np.sum(psd)
        
        if total_power > 0:
            # Mean frequency
            mean_freq = np.sum(freqs * psd) / total_power
            
            # Median frequency
            cum_power = np.cumsum(psd)
            median_idx = np.where(cum_power >= total_power/2)[0]
            median_freq = freqs[median_idx[0]] if len(median_idx) > 0 else 0
            
            # Peak frequency
            peak_idx = np.argmax(psd)
            peak_freq = freqs[peak_idx] if len(psd) > 0 else 0
            
            # Band power ratios
            low_band_mask = (freqs >= 20) & (freqs <= 100)
            med_band_mask = (freqs > 100) & (freqs <= 250)
            high_band_mask = (freqs > 250) & (freqs <= 450)
            
            low_band_power = np.sum(psd[low_band_mask]) / total_power if np.any(low_band_mask) else 0
            med_band_power = np.sum(psd[med_band_mask]) / total_power if np.any(med_band_mask) else 0
            high_band_power = np.sum(psd[high_band_mask]) / total_power if np.any(high_band_mask) else 0
            
            return mean_freq, median_freq, peak_freq, low_band_power, med_band_power, high_band_power
        else:
            return 0, 0, 0, 0, 0, 0
            
    except Exception as e:
        print(f"Welch spectral analysis failed: {e}")
        return 0, 0, 0, 0, 0, 0

def visualize_movement_metrics(metrics, output_prefix="Movement_metrics/movement_metrics"):
    """
    Visualize movement-level metrics with better error handling
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set matplotlib backend to avoid display issues
    matplotlib.use('Agg')
    
    # Check if we have enough data to visualize
    if metrics['true_movement_count'] == 0:
        print(f"Warning: No movement events to visualize")
        
        # Create a simple summary table anyway
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        
        table_data = [
            ['Metric', 'Value'],
            ['Total True Movements', f"{metrics['true_movement_count']}"],
            ['Detected Movements', f"{metrics['detected_movement_count']}"],
            ['Overall Detection Rate', f"{metrics['detection_rate']:.2f}"],
            ['False Activation Rate', f"{metrics['false_activation_rate']:.2f} per minute"]
        ]
        
        table = ax.table(cellText=table_data, colWidths=[0.5, 0.5], loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        plt.title('Movement-Level Metrics Summary (No Events Detected)', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_summary.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        return
    
    # 1. Detection Rate by Movement Type
    if metrics['per_movement_metrics']:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            movements = list(metrics['per_movement_metrics'].keys())
            detection_rates = [metrics['per_movement_metrics'][m]['detection_rate'] for m in movements]
            
            bars = ax.bar(movements, detection_rates, color=['blue', 'green', 'red', 'orange', 'purple'][:len(movements)])
            ax.set_title('Movement Detection Rate by Movement Type')
            ax.set_ylabel('Detection Rate')
            ax.set_ylim(0, 1.05)
            
            # Add values on top of bars
            for bar, rate in zip(bars, detection_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{rate:.2f}', ha='center', fontweight='bold')
            
            # Add overall detection rate as horizontal line
            ax.axhline(y=metrics['detection_rate'], color='black', linestyle='--', 
                      label=f'Overall: {metrics["detection_rate"]:.2f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{output_prefix}_detection_rates.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error creating detection rates plot: {e}")
    
    # 2. Detection Latency Distribution
    if metrics['detection_latencies']:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use matplotlib histogram instead of seaborn to avoid pandas issues
            ax.hist(metrics['detection_latencies'], bins=20, alpha=0.7, density=True)
            
            ax.axvline(metrics['mean_detection_latency'], color='red', linestyle='--',
                      label=f'Mean: {metrics["mean_detection_latency"]:.1f} ms')
            
            if 'median_detection_latency' in metrics:
                ax.axvline(metrics['median_detection_latency'], color='green', linestyle='-.',
                          label=f'Median: {metrics["median_detection_latency"]:.1f} ms')
            
            ax.set_title('Detection Latency Distribution')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Density')
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{output_prefix}_latency_distribution.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error creating latency distribution plot: {e}")
        
        # 3. Latency by Movement Type
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            movements_with_latencies = []
            mean_latencies = []
            
            for movement, movement_metrics in metrics['per_movement_metrics'].items():
                if 'mean_latency' in movement_metrics:
                    movements_with_latencies.append(movement)
                    mean_latencies.append(movement_metrics['mean_latency'])
            
            if movements_with_latencies:
                bars = ax.bar(movements_with_latencies, mean_latencies, 
                             color=['blue', 'green', 'red', 'orange', 'purple'][:len(movements_with_latencies)])
                ax.set_title('Mean Detection Latency by Movement Type')
                ax.set_ylabel('Latency (ms)')
                
                # Add values on top of bars
                for bar, latency in zip(bars, mean_latencies):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                           f'{latency:.1f}', ha='center', fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(f'{output_prefix}_latency_by_movement.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Error creating latency by movement plot: {e}")
    
    # 4. Summary metrics in a table
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Create summary table
        table_data = [
            ['Metric', 'Value'],
            ['Total True Movements', f"{metrics['true_movement_count']}"],
            ['Detected Movements', f"{metrics['detected_movement_count']}"],
            ['Overall Detection Rate', f"{metrics['detection_rate']:.2f}"],
            ['False Activation Rate', f"{metrics['false_activation_rate']:.2f} per minute"]
        ]
        
        # Add latency metrics if available
        if 'mean_detection_latency' in metrics and metrics['mean_detection_latency'] > 0:
            table_data.append(['Mean Detection Latency', f"{metrics['mean_detection_latency']:.1f} ms"])
        else:
            table_data.append(['Mean Detection Latency', "N/A"])
            
        if 'median_detection_latency' in metrics and metrics['median_detection_latency'] > 0:
            table_data.append(['Median Detection Latency', f"{metrics['median_detection_latency']:.1f} ms"])
        else:
            table_data.append(['Median Detection Latency', "N/A"])
        
        # Add per-movement metrics
        for movement, movement_metrics in metrics['per_movement_metrics'].items():
            detection_rate = movement_metrics.get('detection_rate', 0)
            true_count = movement_metrics.get('true_count', 0)
            detected_count = movement_metrics.get('detected_count', 0)
            
            table_data.append([
                f"{movement} Detection Rate", 
                f"{detection_rate:.2f} ({detected_count}/{true_count})"
            ])
            
            if 'mean_latency' in movement_metrics:
                table_data.append([
                    f"{movement} Mean Latency",
                    f"{movement_metrics['mean_latency']:.1f} ms"
                ])
        
        # Create the table
        table = ax.table(cellText=table_data, colWidths=[0.5, 0.5], loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        plt.title('Movement-Level Metrics Summary', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_summary.png', dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error creating summary table: {e}")
    
    print(f"Movement metrics visualization saved with prefix '{output_prefix}'")


def evaluate_realistic_movement_metrics(results, classifier_name, true_labels, pred_labels, confidences=None, window_size=500, overlap=0.75, sampling_rate=2000, min_duration_seconds=0.5, confidence_threshold=0.6, output_prefix=None):
    if output_prefix is None:
        output_prefix = f"Movement_metrics/realistic_movement_metrics_{classifier_name}"
    
    # DON'T filter out Transition/Unknown - just skip them during event detection
    # but preserve them in the sequence to maintain temporal context
    
    print(f"\n--- Realistic Movement-Level Metrics for {classifier_name} ---")
    print(f"Using minimum duration: {min_duration_seconds}s, confidence threshold: {confidence_threshold}")
    
    movement_metrics = calculate_realistic_movement_metrics(
        true_labels, pred_labels, confidences,
        window_size=window_size, overlap=overlap, sampling_rate=sampling_rate,
        min_duration_seconds=min_duration_seconds, confidence_threshold=confidence_threshold
    )
    
    # Rest of function remains the same...
    return movement_metrics


def train_evaluate_with_temporal_smoothing(features, labels, groups, subject_id=None, label_names=None, 
                                         window_size=5, min_confidence=0.6, feature_names=None):
    if len(features) == 0:
        print("Error: No features to train on")
        return {}
    
    # Ensure labels and groups are numpy arrays
    labels_array = np.array(labels)
    groups_array = np.array(groups)
    
    if label_names is None:
        unique_labels = np.unique(labels_array)
        label_names = [str(label) for label in unique_labels]
    
    # Use GroupKFold for cross-validation based on trial groups
    cv = GroupKFold(n_splits=len(np.unique(groups_array)))
    
    # Define classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', gamma='scale', C=1.0, probability=True),
        'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=50,        # Reduced from 100
            learning_rate=0.2,      # Increased from 0.1 for faster convergence
            max_depth=3,            # Reduced from default 3 (or whatever you had)
            subsample=0.8,          # Add subsampling for speed
            random_state=42
        ),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    # Initialize results
    results = {}
    best_accuracy = 0
    best_classifier = None
    
    # Function for temporal smoothing
    def apply_smoothing(predictions, probabilities=None):
        smoothed = []
        n_preds = len(predictions)
        
        for i in range(n_preds):
            # Get window of predictions
            start = max(0, i - window_size + 1)
            window_preds = predictions[start:i+1]
            
            # Use majority voting
            unique_preds, counts = np.unique(window_preds, return_counts=True)
            smoothed_pred = unique_preds[np.argmax(counts)]
            
            # If probabilities provided, check confidence
            if probabilities is not None and i > 0:
                # Only change state if confidence is sufficient
                curr_prob = np.max(probabilities[i])
                prev_pred = smoothed[i-1] if i > 0 else predictions[0]
                
                if smoothed_pred != prev_pred and curr_prob < min_confidence:
                    smoothed_pred = prev_pred  # Keep previous state if confidence is low
            
            smoothed.append(smoothed_pred)
        
        return np.array(smoothed)
    
    # Train and evaluate each classifier
    for name, classifier in classifiers.items():
        print(f"\n--- Training {name} Classifier ---")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])
        
        # Storage for results
        raw_true = []
        raw_pred = []
        smoothed_true = []
        smoothed_pred = []
        fold_accuracies_raw = []
        fold_accuracies_smoothed = []
        
        # Perform cross-validation
        for i, (train_idx, test_idx) in enumerate(cv.split(features, labels_array, groups_array)):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels_array[train_idx], labels_array[test_idx]
            test_group = groups_array[test_idx[0]]
            
            # Remove "Transition" and "Unknown" labels from training
            valid_idx = np.where((y_train != "Transition") & (y_train != "Unknown"))[0]
            X_train_valid = X_train[valid_idx]
            y_train_valid = y_train[valid_idx]
            
            # Train classifier
            pipeline.fit(X_train_valid, y_train_valid)
            
            # Make predictions
            y_pred_raw = pipeline.predict(X_test)
            
            # Get probabilities if available
            if hasattr(classifier, "predict_proba"):
                probabilities = pipeline.predict_proba(X_test)
            else:
                probabilities = None
            
            # Apply temporal smoothing
            y_pred_smoothed = apply_smoothing(y_pred_raw, probabilities)
            
            # Store results
            raw_true.extend(y_test)
            raw_pred.extend(y_pred_raw)
            smoothed_true.extend(y_test)
            smoothed_pred.extend(y_pred_smoothed)
            
            # Calculate fold accuracy (excluding Transition and Unknown)
            valid_test_idx = np.where((y_test != "Transition") & (y_test != "Unknown"))[0]
            if len(valid_test_idx) > 0:
                acc_raw = accuracy_score(y_test[valid_test_idx], y_pred_raw[valid_test_idx])
                acc_smoothed = accuracy_score(y_test[valid_test_idx], y_pred_smoothed[valid_test_idx])
                fold_accuracies_raw.append(acc_raw)
                fold_accuracies_smoothed.append(acc_smoothed)
                print(f"  Fold {i+1} (Group {test_group}): Raw Accuracy = {acc_raw:.4f}, Smoothed = {acc_smoothed:.4f}")
        
        # Calculate overall performance (excluding Transition and Unknown)
        valid_idx = np.where((np.array(raw_true) != "Transition") & (np.array(raw_true) != "Unknown"))[0]
        if len(valid_idx) > 0:
            acc_raw = accuracy_score(np.array(raw_true)[valid_idx], np.array(raw_pred)[valid_idx])
            acc_smoothed = accuracy_score(np.array(smoothed_true)[valid_idx], np.array(smoothed_pred)[valid_idx])
            
            print(f"  Overall Raw Accuracy: {acc_raw:.4f}")
            print(f"  Overall Smoothed Accuracy: {acc_smoothed:.4f}")
            print(f"  Fold Raw Accuracies: {fold_accuracies_raw}")
            print(f"  Fold Smoothed Accuracies: {fold_accuracies_smoothed}")
            print(f"  Avg Raw Accuracy: {np.mean(fold_accuracies_raw):.4f} ± {np.std(fold_accuracies_raw):.4f}")
            print(f"  Avg Smoothed Accuracy: {np.mean(fold_accuracies_smoothed):.4f} ± {np.std(fold_accuracies_smoothed):.4f}")
        
            # Classification report
            print("\n--- Classification Report (Smoothed) ---")
            valid_movements = [label for label in label_names if label not in ["Transition", "Unknown"]]
            report = classification_report(
                np.array(smoothed_true)[valid_idx], 
                np.array(smoothed_pred)[valid_idx],
                target_names=valid_movements,
                output_dict=True  # This returns a dictionary instead of string
            )
            print(classification_report(
                np.array(smoothed_true)[valid_idx], 
                np.array(smoothed_pred)[valid_idx],
                target_names=valid_movements
            ))

            
            # Confusion matrix
            cm = confusion_matrix(
                np.array(smoothed_true)[valid_idx], 
                np.array(smoothed_pred)[valid_idx],
                labels=valid_movements
            )
            print("Confusion Matrix:")
            print(cm)
            
            # Store results
            results[name] = {'pipeline': pipeline,'raw_accuracy': acc_raw,'smoothed_accuracy': acc_smoothed, 'fold_accuracies_raw': fold_accuracies_raw,
                'fold_accuracies_smoothed': fold_accuracies_smoothed,'avg_accuracy_raw': np.mean(fold_accuracies_raw),'avg_accuracy_smoothed': np.mean(fold_accuracies_smoothed),
                'std_accuracy_raw': np.std(fold_accuracies_raw),'std_accuracy_smoothed': np.std(fold_accuracies_smoothed),'confusion_matrix': cm,
                'raw_predictions': np.array(raw_pred),'smoothed_predictions': np.array(smoothed_pred),'true_labels': np.array(raw_true),'classification_report': report
            }

            if name in ["GradientBoosting","RandomForest"]: # Add other tree models if used
                print(f"\nCalculating Feature Importances for {name}...")
                # Create a fresh pipeline instance for this full training run
                # to avoid using the one last trained on a CV fold.
                final_training_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    # Re-instantiate the classifier to ensure it's fresh
                    ('classifier', classifiers[name]) # Get a fresh instance from your dict
                ])

                all_valid_train_idx_for_final_model = np.where((labels_array != "Transition") & (labels_array != "Unknown"))[0]
                X_all_train_final = features[all_valid_train_idx_for_final_model]
                y_all_train_final = labels_array[all_valid_train_idx_for_final_model]
                
                final_training_pipeline.fit(X_all_train_final, y_all_train_final)
                
                # Access the fitted classifier step
                fitted_classifier_for_importance = final_training_pipeline.named_steps['classifier']
                
                if hasattr(fitted_classifier_for_importance, 'feature_importances_'):
                    importances = fitted_classifier_for_importance.feature_importances_
                    
                    feature_names_for_importance = feature_names # Use the passed list
                    
                    if len(feature_names_for_importance) == len(importances):
                        feature_importance_df = pd.DataFrame({
                            'feature': feature_names_for_importance, 
                            'importance': importances
                        })
                        feature_importance_df = feature_importance_df.sort_values(
                            by='importance', ascending=False
                        )
                        print(f"\nTop 20 Feature Importances ({name}):")
                        print(feature_importance_df.head(20))

                        plt.figure(figsize=(12, max(8, len(feature_names_for_importance) // 3))) # Adjusted size
                        sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20), palette="viridis")
                        plt.title(f'Top 20 Feature Importances - {name}')
                        plt.tight_layout()
                        # Ensure Results directory exists
                        os.makedirs("Results", exist_ok=True)
                        plt.savefig(f'Results/feature_importances_{name}.png')
                        plt.close()
                        
                        results[name]['feature_importances'] = importances # Store them
                    else:
                        print(f"ERROR: Mismatch for {name} between feature names ({len(feature_names_for_importance)}) and importances ({len(importances)})!")
                else:
                    print(f"Classifier {name} does not have 'feature_importances_' attribute.")
            
            # Track best classifier
            if acc_smoothed > best_accuracy:
                best_accuracy = acc_smoothed
                best_classifier = name
            
            # Save model
            if subject_id:
                model_filename = f"model_{name}_{subject_id}_temporal.pkl"
                
                # Save metadata along with model
                model_data = {
                    'pipeline': pipeline,
                    'window_size': window_size,
                    'min_confidence': min_confidence,
                    'label_names': label_names,
                    'history_size': 3  # Default history size
                }
                
                joblib.dump(model_data, model_filename)
                print(f"  Model saved as {model_filename}")
    
    if best_classifier:
        print(f"\nBest classifier: {best_classifier} with smoothed accuracy {best_accuracy:.4f}")

    
    return results

def train_evaluate_with_movement_metrics(features, labels, groups, subject_id=None, label_names=None, 
                                       temporal_smoothing_window_len=5, min_confidence=0.6,
                                       emg_window_len_for_metrics=500, overlap=0.75, 
                                       sampling_rate=2000, feature_names=None):
    # First, perform window-level evaluation
    window_results = train_evaluate_with_temporal_smoothing(
        features, labels, groups, subject_id, label_names, temporal_smoothing_window_len, min_confidence, feature_names
    )
    
    if not window_results:
        return {}
    
    # Then, add movement-level metrics for each classifier
    for name, results in window_results.items():
        print(f"\n--- Adding Movement-Level Metrics for {name} Classifier ---")
        
        # Extract true and predicted labels
        true_labels = results['true_labels']
        smoothed_pred = results['smoothed_predictions']
        
        # Try multiple parameter combinations for movement detection
        print("\n--- Trying different movement detection parameters ---")
        
        # Standard parameters
        print("\n1. Standard parameters:")
        standard_metrics = evaluate_realistic_movement_metrics(
            results, name, true_labels, smoothed_pred,
            window_size=emg_window_len_for_metrics, overlap=overlap, sampling_rate=sampling_rate,
            min_duration_seconds=0.5, confidence_threshold=0.6,
            output_prefix=f"Movement_metrics/movement_metrics_standard_{name}_{subject_id if subject_id else 'combined'}"
        )
        
        # More sensitive parameters
        print("\n2. More sensitive parameters:")
        sensitive_metrics = evaluate_realistic_movement_metrics(
            results, name, true_labels, smoothed_pred,
            window_size=emg_window_len_for_metrics, overlap=overlap, sampling_rate=sampling_rate,
            min_duration_seconds=0.2, confidence_threshold=0.5,
            output_prefix=f"Movement_metrics/movement_metrics_sensitive_{name}_{subject_id if subject_id else 'combined'}"
        )
        
        # Choose the best metrics based on detection rate
        if sensitive_metrics['true_movement_count'] > standard_metrics['true_movement_count']:
            print("\nUsing sensitive parameters (found more movement events)")
            results['movement_metrics'] = sensitive_metrics
        else:
            print("\nUsing standard parameters")
            results['movement_metrics'] = standard_metrics
    
    return window_results


def visualize_class_performance(results, classifier_name, label_names):
    """Visualize per-class performance metrics"""
    # Extract confusion matrix
    cm = results[classifier_name]['confusion_matrix']
    valid_movements = [label for label in label_names if label not in ["Transition", "Unknown"]]
    
    # Calculate per-class metrics from confusion matrix
    n_classes = len(valid_movements)
    class_accuracy = np.zeros(n_classes)
    
    report_dict = results[classifier_name].get('classification_report_dict') # Get the stored dict

    if report_dict: # Check if the dictionary was stored
        class_recalls = []
        for movement in valid_movements:
            if movement in report_dict and isinstance(report_dict[movement], dict): # Check if movement key exists and is a dict
                class_recalls.append(report_dict[movement]['f1-score'])
            else:
                print(f"Warning: Movement '{movement}' not found or not a dict in report_dict for {classifier_name}. Using 0 for recall.")
                class_recalls.append(0.0) 
        class_accuracy = np.array(class_recalls)
    else:
        # Fallback to calculating from CM if report_dict is not available
        print(f"Warning: classification_report_dict not found for {classifier_name}. Calculating from CM.")
        cm = results[classifier_name]['confusion_matrix']
        n_classes = len(valid_movements)
        class_accuracy = np.zeros(n_classes)
        if cm.shape[0] == n_classes and cm.shape[1] == n_classes: # Basic check
            for i in range(n_classes):
                tp = cm[i, i]
                actual_total = np.sum(cm[i, :])
                class_accuracy[i] = tp / actual_total if actual_total > 0 else 0.0
        else:
            print(f"Error: CM shape {cm.shape} does not match n_classes {n_classes} for {classifier_name}.")
            class_accuracy = np.zeros(n_classes) # or handle error appropriately
    
    # Create bar plot for class accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(valid_movements, class_accuracy, color=['green', 'blue', 'red'])
    plt.title(f'Per-Class F1-Score - {classifier_name}')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{acc:.2f}', ha='center', fontweight='bold')
    
    # Save the figure
    plt.savefig(f'Results/class_accuracy_{classifier_name}.png')
    plt.close()
    
    # Visualize confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=valid_movements, yticklabels=valid_movements)
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'Results/confusion_matrix_{classifier_name}.png')
    plt.close()
    
    # Print detailed per-class metrics
    print("\n--- Per-Class Performance Metrics ---")
    print(f"{'Movement':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print('-' * 50)
    
    # Extract true and predicted labels
    y_true = results[classifier_name]['true_labels']
    y_pred = results[classifier_name]['smoothed_predictions']
    
    # Only consider valid movements (not Transition or Unknown)
    valid_idx = np.where((y_true != "Transition") & (y_true != "Unknown"))[0]
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]
    
    # Calculate metrics for each class
    for i, movement in enumerate(valid_movements):
        # True instances of this movement
        class_indices = (y_true == movement)
        # True positive rate (recall)
        recall = np.mean(y_pred[class_indices] == movement) if np.any(class_indices) else 0
        # Precision
        pred_indices = (y_pred == movement)
        precision = np.mean(y_true[pred_indices] == movement) if np.any(pred_indices) else 0
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{movement:<10} {class_accuracy[i]:.4f}     {precision:.4f}     {recall:.4f}     {f1:.4f}")

def save_f1_scores_table(results, movements, output_filename="Results/f1_scores_table.csv"):
    """
    Save F1 scores for each classifier and movement to a CSV table
    """
    import pandas as pd
    
    # Initialize the table
    f1_data = []
    
    # Extract F1 scores for each classifier
    for classifier_name, classifier_results in results.items():
        row = {'Classifier': classifier_name}
        
        # Get classification report
        if 'classification_report' in classifier_results:
            report = classifier_results['classification_report']
            
            # Extract F1 scores for each movement
            for movement in movements:
                if movement in ["Transition", "Unknown"]:
                    continue  # Skip these labels
                    
                if movement in report and isinstance(report[movement], dict):
                    f1_score = report[movement].get('f1-score', 0.0)
                else:
                    f1_score = 0.0
                
                row[f'{movement}_F1'] = f1_score
            
            # Add overall metrics
            if 'macro avg' in report and isinstance(report['macro avg'], dict):
                row['Macro_Avg_F1'] = report['macro avg'].get('f1-score', 0.0)
            else:
                row['Macro_Avg_F1'] = 0.0
                
            if 'weighted avg' in report and isinstance(report['weighted avg'], dict):
                row['Weighted_Avg_F1'] = report['weighted avg'].get('f1-score', 0.0)
            else:
                row['Weighted_Avg_F1'] = 0.0
            
            # Add accuracy
            row['Accuracy'] = classifier_results.get('smoothed_accuracy', 0.0)
            
        f1_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(f1_data)
    df = df.round(4)  # Round to 4 decimal places
    
    # Sort by Macro_Avg_F1 (best first)
    df = df.sort_values('Macro_Avg_F1', ascending=False)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    df.to_csv(output_filename, index=False)
    
    print(f"\nF1 scores table saved to: {output_filename}")
    print("\nF1 Scores Summary:")
    print(df.to_string(index=False))
    
    return df

def save_movement_metrics_table(results, output_filename="Results/movement_metrics_table.csv"):
    """
    Save movement-level metrics for each classifier to a CSV table
    """
    import pandas as pd
    
    metrics_data = []
    
    for classifier_name, classifier_results in results.items():
        if 'realistic_movement_metrics' in classifier_results:
            metrics = classifier_results['realistic_movement_metrics']
            
            row = {
                'Classifier': classifier_name,
                'Detection_Rate': metrics.get('detection_rate', 0.0),
                'False_Activation_Rate': metrics.get('false_activation_rate', 0.0),
                'Mean_Latency_ms': metrics.get('mean_detection_latency', 0.0),
                'Median_Latency_ms': metrics.get('median_detection_latency', 0.0),
                'True_Movement_Count': metrics.get('true_movement_count', 0),
                'Detected_Movement_Count': metrics.get('detected_movement_count', 0)
            }
            
            # Add per-movement detection rates
            per_movement = metrics.get('per_movement_metrics', {})
            for movement, movement_metrics in per_movement.items():
                row[f'{movement}_Detection_Rate'] = movement_metrics.get('detection_rate', 0.0)
                row[f'{movement}_Mean_Latency_ms'] = movement_metrics.get('mean_latency', 0.0)
            
            metrics_data.append(row)
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        df = df.round(4)
        
        # Sort by Detection_Rate (best first)
        df = df.sort_values('Detection_Rate', ascending=False)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        df.to_csv(output_filename, index=False)
        
        print(f"\nMovement metrics table saved to: {output_filename}")
        print("\nMovement Metrics Summary:")
        print(df.to_string(index=False))
        
        return df
    else:
        print("No movement metrics found to save")
        return None



def select_best_model_for_realtime(model_results, min_duration=0.5, emg_window_size=500, 
                                 confidence_threshold=0.6, custom_weights=None):
    """
    Select the best classifier based on realistic movement metrics - FIXED VERSION
    """
    # FIXED: Proper parameter handling - ensure scalar values
    if isinstance(min_duration, dict):
        print(f"WARNING: min_duration is a dictionary: {min_duration}")
        min_duration_val = 0.5  # Default value
    else:
        try:
            min_duration_val = float(min_duration)
        except (TypeError, ValueError):
            print(f"WARNING: Could not convert min_duration to float: {min_duration}")
            min_duration_val = 0.5
    
    if isinstance(confidence_threshold, dict):
        print(f"WARNING: confidence_threshold is a dictionary: {confidence_threshold}")
        confidence_threshold_val = 0.6  # Default value
    else:
        try:
            confidence_threshold_val = float(confidence_threshold)
        except (TypeError, ValueError):
            print(f"WARNING: Could not convert confidence_threshold to float: {confidence_threshold}")
            confidence_threshold_val = 0.6
    
    # FIXED: Ensure emg_window_size is also properly handled
    try:
        emg_window_size_val = int(emg_window_size)
    except (TypeError, ValueError):
        print(f"WARNING: Could not convert emg_window_size to int: {emg_window_size}")
        emg_window_size_val = 500
    
    # Use default weights if none provided
    if custom_weights is None:
        custom_weights = {
            'detection_rate': 0.4,
            'false_activation_penalty': 0.4,
            'latency_penalty': 0.2
        }
    
    all_scores = {}
    print(f"\n--- Selecting Best Model for Real-time Applications ---")
    print(f"Using min_duration={min_duration_val}s, confidence={confidence_threshold_val}")
    
    for name, results in model_results.items():
        # Extract true and predicted labels
        true_labels = results['true_labels']
        pred_labels = results['smoothed_predictions']
        
        # Get confidence values if available
        confidences = get_prediction_confidences(results, true_labels)
        
        # Calculate realistic movement metrics with FIXED parameters
        print(f"\nEvaluating {name} Classifier...")
        
        try:
            movement_metrics = calculate_realistic_movement_metrics(
                true_labels, pred_labels, confidences, 
                window_size=emg_window_size_val,
                min_duration_seconds=min_duration_val,
                confidence_threshold=confidence_threshold_val
            )
            
            # Create output directory if it doesn't exist
            os.makedirs("Movement_metrics", exist_ok=True)
            
            # FIX: Add error handling for visualization
            try:
                visualize_movement_metrics(
                    movement_metrics, output_prefix=f"Movement_metrics/movement_metrics_{name}"
                )
            except Exception as viz_error:
                    print(f"Warning: Visualization failed for {name}: {viz_error}")
                    # Continue without visualization
                    
        except Exception as e:
            print(f"ERROR calculating movement metrics for {name}: {e}")
            import traceback
            traceback.print_exc()  # This will show the full error
            movement_metrics = {
                'true_movement_count': 0,
                'detected_movement_count': 0,
                'detection_rate': 0,
                'false_activation_rate': 0,
                'mean_detection_latency': 0,
                'per_movement_metrics': {}
        }
        
        # Calculate composite score
        score = calculate_movement_composite_score(movement_metrics, custom_weights)
        all_scores[name] = score
        
        print(f"{name} Real-time Performance:")
        print(f"  Movement Detection Rate: {movement_metrics.get('detection_rate', 0):.4f}")
        print(f"  False Activation Rate: {movement_metrics.get('false_activation_rate', 0):.4f} per minute")
        latency = movement_metrics.get('mean_detection_latency', 0)
        if latency > 0:
            print(f"  Mean Detection Latency: {latency:.2f} ms")
        print(f"  Composite Score: {score:.4f}")
        
        # Store the metrics in results
        results['realistic_movement_metrics'] = movement_metrics
    
    # Find best classifier
    if all_scores:
        best_classifier = max(all_scores, key=all_scores.get)
        best_score = all_scores[best_classifier]
        print(f"\nBest classifier for real-time use: {best_classifier} with score {best_score:.4f}")
    else:
        print("Warning: No scores calculated for any classifier")
        best_classifier = None
        best_score = 0
    
    return best_classifier, best_score, all_scores

def get_prediction_confidences(results, true_labels):
    """
    Extract or generate prediction confidences
    """
    try:
        if 'pipeline' in results:
            pipeline = results['pipeline']
            if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                # Would need to recalculate with test data, for now return None
                return None
        return None
    except Exception as e:
        print(f"Warning: Error getting confidences: {e}")
        return None

def calculate_movement_composite_score(movement_metrics, weights):
    """
    Calculate composite score from movement metrics
    """
    detection_rate = movement_metrics.get('detection_rate', 0)
    
    false_rate = movement_metrics.get('false_activation_rate', 0)
    max_false_rate = 3.0
    false_activation_penalty = 1 - min(false_rate / max_false_rate, 1.0)
    
    latency = movement_metrics.get('mean_detection_latency', 0)
    max_latency = 500.0
    latency_penalty = 1 - min(latency / max_latency, 1.0) if latency > 0 else 1.0
    
    score = (
        weights['detection_rate'] * detection_rate +
        weights['false_activation_penalty'] * false_activation_penalty +
        weights['latency_penalty'] * latency_penalty
    )
    
    return score


def detect_single_movement(predictions, confidences, window_buffer_size=20, detection_threshold=5):
    """
    Detect a single movement using a window-based approach, optimized for the one-movement scenario
    
    Args:
        predictions: Array of movement predictions
        confidences: Array of confidence values
        window_buffer_size: Size of the sliding window
        detection_threshold: Minimum number of windows with same prediction to confirm movement
    
    Returns:
        detected_movement: The detected movement type
        movement_confidence: Confidence score for the detected movement
    """
    # Initialize counts for each movement type
    movement_counts = {}
    
    # Count only the movements with sufficient confidence
    for pred, conf in zip(predictions, confidences):
        if conf >= 0.5 and pred != "Rest":
            movement_counts[pred] = movement_counts.get(pred, 0) + 1
    
    # If no movements detected with sufficient confidence, return Rest
    if not movement_counts:
        return "Rest", 1.0
    
    # Find movement with highest count
    detected_movement = max(movement_counts.items(), key=lambda x: x[1])
    movement_type, count = detected_movement
    
    # Only return a movement if it exceeds the threshold
    if count >= detection_threshold:
        # Calculate confidence as proportion of windows with this movement
        confidence = count / window_buffer_size
        return movement_type, confidence
    else:
        return "Rest", 1.0
    

def plot_activity_detection_and_window_labels(
    time_vector,
    smoothed_energy,
    amplitude_threshold,
    activity_mask,
    window_center_samples,
    window_final_labels,
    movement_label_for_trial,
    sampling_rate,
    output_filename="activities/activity_plot.png"
):
    """
    Visualizes the smoothed EMG energy, threshold, detected activity,
    and the labels assigned to segmented windows.

    Args:
        time_vector: Time vector for the smoothed_energy signal (in seconds).
        smoothed_energy: The smoothed energy signal.
        amplitude_threshold: The calculated amplitude threshold.
        activity_mask: Boolean array indicating detected active regions.
        window_center_samples: List of sample indices for the center of each window.
        window_final_labels: List of final labels assigned to each window.
        movement_label_for_trial: The true label of the trial being processed.
        sampling_rate: Sampling rate of the original EMG.
        output_filename: Name of the file to save the plot.
    """
    fig, ax1 = plt.subplots(figsize=(15, 7))

    # Plot smoothed energy
    color_energy = 'tab:blue'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Smoothed Energy', color=color_energy)
    ax1.plot(time_vector, smoothed_energy, color=color_energy, label='Smoothed Energy')
    ax1.tick_params(axis='y', labelcolor=color_energy)
    ax1.axhline(amplitude_threshold, color='r', linestyle='--', label=f'Amplitude Threshold ({amplitude_threshold:.2e})')

    # Shade active regions based on activity_mask
    ax1.fill_between(time_vector, 0, ax1.get_ylim()[1], where=activity_mask,
                     color='green', alpha=0.3, label='Detected Activity')

    # Create a second y-axis for window labels (if any windows exist)
    if window_center_samples:
        ax2 = ax1.twinx()
        color_labels = 'tab:red'
        ax2.set_ylabel('Window Labels', color=color_labels) # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=color_labels)
        ax2.set_ylim(-0.5, len(np.unique(window_final_labels)) - 0.5) # Adjust based on number of unique labels

        # Map labels to numerical values for plotting
        unique_final_labels = sorted(list(set(window_final_labels)))
        label_to_int = {label: i for i, label in enumerate(unique_final_labels)}
        window_labels_int = [label_to_int[l] for l in window_final_labels]
        window_center_times = [s / sampling_rate for s in window_center_samples]

        # Plot window labels as scatter points at their center times
        for label_str, label_int_val in label_to_int.items():
            # Find indices for the current label
            current_label_indices = [i for i, l_int in enumerate(window_labels_int) if l_int == label_int_val]
            if current_label_indices:
                # Get the times and y-values for these points
                current_label_times = [window_center_times[i] for i in current_label_indices]
                current_label_y_values = [window_labels_int[i] for i in current_label_indices]
                ax2.scatter(current_label_times, current_label_y_values,
                            label=f'Window: {label_str}', alpha=0.7, marker='o', s=50)


        ax2.set_yticks(list(label_to_int.values()))
        ax2.set_yticklabels(list(label_to_int.keys()))
    else:
        print("No windows to plot for labels.")


    fig.suptitle(f'Activity Detection & Window Labeling for Trial: {movement_label_for_trial}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    if window_center_samples: # Only add ax2 legend if it exists
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')


    plt.savefig(output_filename)
    print(f"Saved activity plot to {output_filename}")
    plt.close(fig)

def generate_complete_feature_names(n_channels, basic_features, include_context=True):
    """
    Generate complete feature names including cross-channel correlations - FIXED VERSION
    """
    feature_names = []
    
    # 1. Basic features per channel
    for c in range(n_channels):
        channel_prefix = f"Ch{c+1}_"
        for f_name in basic_features:
            feature_names.append(f"{channel_prefix}{f_name}")
    
    # 2. Cross-channel correlation features (FIXED: was missing)
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            feature_names.append(f"CrossCorr_Ch{i+1}_Ch{j+1}")
    
    # 3. Ratio features (conditional based on channel count)
    if n_channels >= 2:
        # Biceps/Triceps ratio
        triceps_ch = 2 if n_channels == 2 else 3
        feature_names.append(f"Ratio_Ch1_Ch{triceps_ch}_MAV")
    
    if n_channels == 4:
        # FDS/EDC ratio (assuming FDS=Ch2, EDC=Ch4)
        feature_names.append("Ratio_Ch2_Ch4_MAV")
    
    # 4. Delta features (if context is included)
    if include_context:
        base_feature_count = len(feature_names)
        delta_names = [f"{name}_Delta" for name in feature_names[:base_feature_count]]
        feature_names.extend(delta_names)
    
    return feature_names

# Updated usage in main function
def get_feature_names_for_data(n_channels):
    """
    Get properly ordered feature names that match extract_features_with_context output
    """
    basic_features = [
        "MAV", "RMS", "ZC", "WL", "Entropy", "Variance",
        "MeanFreq", "MedianFreq", "PeakFreq", 
        "LowBandPower", "MedBandPower", "HighBandPower",
        "RMS_Low", "RMS_Mid", "RMS_High",  # Enhanced RMS features
    ]
    
    return generate_complete_feature_names(n_channels, basic_features, include_context=True)
                         

# Add these functions after your existing preprocessing functions

def create_realistic_trial_from_movement_data(movement_data, movement_metadata, 
                                            rest_data_pool, movement_label,
                                            subject_id, trial_id,  # ADD these parameters
                                            pre_rest_duration=1.0, post_rest_duration=1.0):
    """
    Create realistic trials by adding rest periods before and after movements
    """
    sampling_rate = movement_metadata['sampling_rate']
    
    # Calculate samples needed for rest periods
    pre_rest_samples = int(pre_rest_duration * sampling_rate)
    post_rest_samples = int(post_rest_duration * sampling_rate)
    
    # CHANGE: Find subject-matched rest data
    if rest_data_pool:
        # Try to find rest from same subject
        subject_rest_trials = []
        for rest_data, rest_subject, rest_trial in rest_data_pool:
            if rest_subject == subject_id:
                subject_rest_trials.append(rest_data)
        
        if subject_rest_trials:
            # Use rest from same subject
            rest_idx = np.random.randint(0, len(subject_rest_trials))
            rest_trial = subject_rest_trials[rest_idx]
            print(f"Using rest from same subject {subject_id}")
        else:
            # Fallback to any rest if no subject-matched rest available
            rest_idx = np.random.randint(0, len(rest_data_pool))
            rest_trial = rest_data_pool[rest_idx][0]  # Get data from tuple
            print(f"Warning: No rest data for subject {subject_id}, using random rest")
        
        # Extract segments for pre and post rest
        if rest_trial.shape[1] >= (pre_rest_samples + post_rest_samples):
            pre_rest_segment = rest_trial[:, :pre_rest_samples]
            post_rest_segment = rest_trial[:, -post_rest_samples:]
        else:
            # If rest trial too short, repeat it
            rest_repeated = np.tile(rest_trial, (1, 3))  # Repeat 3 times
            pre_rest_segment = rest_repeated[:, :pre_rest_samples]
            post_rest_segment = rest_repeated[:, -post_rest_samples:]
    else:
        # CHANGE: Use synthetic rest based on movement baseline
        print(f"No rest data available, generating synthetic rest for subject {subject_id}")
        n_channels = movement_data.shape[0]
        movement_baseline = np.percentile(movement_data, 5, axis=1, keepdims=True)  # 5th percentile per channel
        movement_noise_level = np.std(movement_data, axis=1, keepdims=True) * 0.1   # 10% of movement variability
        
        pre_rest_segment = np.random.normal(
            movement_baseline, movement_noise_level, (n_channels, pre_rest_samples)
        )
        post_rest_segment = np.random.normal(
            movement_baseline, movement_noise_level, (n_channels, post_rest_samples)
        )
    
    # Combine segments
    realistic_trial = np.concatenate([
        pre_rest_segment,
        movement_data,
        post_rest_segment
    ], axis=1)
    
    return realistic_trial, movement_metadata

def create_realistic_dataset(all_data, all_labels, all_metadata, all_trials, all_subjects,
                           pre_rest_duration=1.0, post_rest_duration=1.0):
    """
    Convert your current dataset to realistic trials with rest periods
    """
    # CHANGE: Store rest data with subject info
    rest_data_pool = []
    for i, label in enumerate(all_labels):
        if label == "Rest":
            rest_data_pool.append((all_data[i], all_subjects[i], all_trials[i]))  # Store (data, subject, trial)
    
    print(f"Found {len(rest_data_pool)} rest trials for realistic trial creation")
    
    realistic_data = []
    realistic_labels = []
    realistic_metadata = []
    realistic_trials = []
    realistic_subjects = []
    
    for i, (data, label, metadata, trial, subject) in enumerate(
        zip(all_data, all_labels, all_metadata, all_trials, all_subjects)):
        
        if label == "Rest":
            # Keep rest trials as-is (they're already realistic)
            realistic_data.append(data)
            realistic_labels.append(label)
            realistic_metadata.append(metadata)
            realistic_trials.append(trial)
            realistic_subjects.append(subject)
        else:
            # CHANGE: Pass subject and trial info
            real_trial, real_metadata = create_realistic_trial_from_movement_data(
                data, metadata, rest_data_pool, label,
                subject, trial,  # ADD these parameters
                pre_rest_duration, post_rest_duration
            )
            
            realistic_data.append(real_trial)
            realistic_labels.append(label)  # Keep original movement label for trial
            realistic_metadata.append(real_metadata)
            realistic_trials.append(trial)
            realistic_subjects.append(subject)
    
    return realistic_data, realistic_labels, realistic_metadata, realistic_trials, realistic_subjects

def create_realistic_window_labels_for_trial(data, label, metadata, window_size=500, overlap=0.75,
                                           pre_rest_duration=1.0, post_rest_duration=1.0):
    """
    Create realistic window labels that match the realistic trial structure
    """
    sampling_rate = metadata['sampling_rate']
    
    # Calculate number of samples for each section
    pre_rest_samples = int(pre_rest_duration * sampling_rate)
    post_rest_samples = int(post_rest_duration * sampling_rate)
    
    # Calculate window parameters
    stride = int(window_size * (1 - overlap))
    total_samples = data.shape[1]
    
    # Calculate approximate number of windows for each section
    pre_rest_windows = max(1, pre_rest_samples // stride)
    post_rest_windows = max(1, post_rest_samples // stride)
    
    # Calculate total windows
    segmented = segment_data(data, window_size, overlap)
    total_windows = segmented.shape[0]
    
    # Create labels: [Rest] -> [Movement] -> [Rest]
    movement_windows = total_windows - pre_rest_windows - post_rest_windows
    movement_windows = max(1, movement_windows)  # Ensure at least 1 movement window
    
    window_labels = (
        ["Rest"] * pre_rest_windows +
        [label] * movement_windows +
        ["Rest"] * post_rest_windows
    )
    
    # Adjust length to match actual windows
    if len(window_labels) > total_windows:
        window_labels = window_labels[:total_windows]
    elif len(window_labels) < total_windows:
        # Add more movement labels if needed
        window_labels.extend([label] * (total_windows - len(window_labels)))
    
    return window_labels

def main():
    # Parameters
    window_size = 500  # 250ms at 2000Hz
    overlap = 0.75  # 75% overlap
    history_size = 3  # Number of past windows for context
    smoothing_window = 5  # Size of smoothing window
    
    # NEW: Add realistic trial parameters
    use_realistic_trials = True  
    pre_rest_duration = 1.5      
    post_rest_duration = 1.0     
    use_adaptive_baseline = True  

    
    # Define movements to include
    movements=['Rest', 'Flexion', 'Extension', 'Pronation', 'Supination', 'Grasp_Power']
    
    print("Loading EMG dataset...")
    all_data, all_labels, all_trials, all_subjects, all_metadata = load_emg_dataset(movements=movements, remove_fds_channel=False)
    
    if not all_data:
        print("Error: No data was loaded. Exiting.")
        return
    
    # NEW: Convert to realistic trials if enabled
    if use_realistic_trials:
        print("\n=== Converting to Realistic Trials ===")
        all_data, all_labels, all_metadata, all_trials, all_subjects = create_realistic_dataset(
            all_data, all_labels, all_metadata, all_trials, all_subjects,
            pre_rest_duration=pre_rest_duration,
            post_rest_duration=post_rest_duration
        )
        print(f"Converted to {len(all_data)} realistic trials")
    
    # Calculate global baseline stats (same as before)
    all_rest_energies = []
    for i, (data, label, trial, subject, metadata) in enumerate(zip(all_data, all_labels, all_trials, all_subjects, all_metadata)):
        if label == "Rest":
            processed_data = preprocess_emg(data, metadata)
            energy = np.mean(np.square(processed_data), axis=0)
            sampling_rate = metadata['sampling_rate']
            smoothed_energy_rest_trial = calculate_smoothed_energy(energy, sampling_rate)
            all_rest_energies.append(smoothed_energy_rest_trial)

    if not all_rest_energies:
        print("Warning: No 'Rest' trials found to calculate global baseline. Will use per-trial baseline.")
        GLOBAL_BASELINE_STATS = None
    else:
        concatenated_rest_energy = np.concatenate(all_rest_energies)
        GLOBAL_BASELINE_STATS = {
            "mean": np.mean(concatenated_rest_energy),
            "std": np.std(concatenated_rest_energy)
        }
        print(f"Global Baseline Stats: Mean Energy={GLOBAL_BASELINE_STATS['mean']:.4f}, Std Energy={GLOBAL_BASELINE_STATS['std']:.4f}")
    
    print("\nPreparing data with temporal context...")
    all_features = []
    all_window_labels = []
    all_segment_trials = []
    all_segment_subjects = []
    
    for i, (data, label, trial, subject, metadata) in enumerate(zip(all_data, all_labels, all_trials, all_subjects, all_metadata)):
        trial_identifier = f"S{subject}_T{trial}"
        print(f"\nProcessing {label} data from subject {subject}, trial {trial}...")
        
        if use_realistic_trials and label != "Rest":
            # Use realistic window labeling for movement trials
            features = extract_features_with_context(
                segment_data(preprocess_emg(data, metadata), window_size, overlap), 
                history_size
            )
            
            if use_adaptive_baseline:
                # Use adaptive baseline approach
                _, window_labels = prepare_emg_data_with_context(
                    data, label, metadata,
                    trial_info_for_plot=trial_identifier,
                    window_size=window_size,
                    overlap=overlap,
                    history_size=history_size,
                    global_baseline_stats=GLOBAL_BASELINE_STATS,
                    plot_each_trial=True,
                    use_adaptive_baseline=True,      # NEW
                    pre_rest_duration=pre_rest_duration,  # NEW
                    post_rest_duration=post_rest_duration  # NEW
                )
            else:
                # Original realistic labeling
                window_labels = create_realistic_window_labels_for_trial(
                    data, label, metadata, window_size, overlap,
                    pre_rest_duration, post_rest_duration
                )
            
        else:
            # Original processing
            features, window_labels = prepare_emg_data_with_context(
                data, label, metadata,
                trial_info_for_plot=trial_identifier,
                window_size=window_size,
                overlap=overlap,
                history_size=history_size,
                global_baseline_stats=GLOBAL_BASELINE_STATS,
                plot_each_trial=True,
                use_adaptive_baseline=use_adaptive_baseline,      # NEW
                pre_rest_duration=pre_rest_duration,              # NEW
                post_rest_duration=post_rest_duration             # NEW
            )
        
        if features.shape[0] == 0:
            print(f"Warning: No features extracted for trial {trial}")
            continue
            
        print(f"Extracted {features.shape[0]} windows, {features.shape[1]} features per window")
        
        all_features.append(features)
        all_window_labels.extend(window_labels)
        all_segment_trials.extend([trial] * features.shape[0])
        all_segment_subjects.extend([subject] * features.shape[0])
    
    # Rest of your code remains the same...
    if not all_features:
        print("Error: No features extracted from any trial")
        return
        
    combined_features = np.vstack(all_features)
    
    print(f"\nTotal windows: {combined_features.shape[0]}")
    print(f"Features per window: {combined_features.shape[1]}")
    
    # Check labels
    unique_labels = np.unique(all_window_labels)
    print(f"Unique window labels: {unique_labels}")
    
    # Count labels
    label_counts = {}
    for label in all_window_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(all_window_labels)*100:.1f}%)")
    
    
    # FIXED: Generate feature names properly
    n_channels = all_data[0].shape[0]
    feature_names = get_feature_names_for_data(n_channels)
    
    # Verify feature count matches
    if len(feature_names) != combined_features.shape[1]:
        print(f"WARNING: Feature name count ({len(feature_names)}) doesn't match feature count ({combined_features.shape[1]})")
        # Pad or truncate feature names to match
        if len(feature_names) < combined_features.shape[1]:
            additional_names = [f"Feature_{i}" for i in range(len(feature_names), combined_features.shape[1])]
            feature_names.extend(additional_names)
        else:
            feature_names = feature_names[:combined_features.shape[1]]
    
    print(f"Generated {len(feature_names)} feature names")
    
    # Extract unique subjects and movements
    unique_subjects = np.unique(all_segment_subjects)
    valid_movements = [label for label in movements if label not in ["Transition", "Unknown"]]
    
    print(f"Unique subjects: {unique_subjects}")
    print(f"Valid movements: {valid_movements}")
    
    # Create necessary directories
    os.makedirs("Results", exist_ok=True)
    os.makedirs("Models", exist_ok=True)
    os.makedirs("Movement_metrics", exist_ok=True)

    # Train combined model
    print("\n--- Training temporal classifier across all subjects ---")
    combined_model_results = train_evaluate_with_movement_metrics(
        combined_features, all_window_labels, all_segment_trials, 
        subject_id="all_subjects_temporal", 
        label_names=unique_labels,
        temporal_smoothing_window_len=smoothing_window,
        emg_window_len_for_metrics=window_size,
        overlap=overlap,
        sampling_rate=2000,
        feature_names=feature_names,  # Use the corrected feature names
    )

    # NEW: Save F1 scores and movement metrics tables
    print("\n--- Saving Performance Tables ---")
    valid_movements = [m for m in movements if m not in ["Transition", "Unknown"]]
    
    # Save F1 scores table
    f1_df = save_f1_scores_table(combined_model_results, valid_movements)
    
    # Save movement metrics table
    movement_df = save_movement_metrics_table(combined_model_results)
    
    # Create a combined summary table
    if f1_df is not None and movement_df is not None:
        summary_df = f1_df.merge(movement_df[['Classifier', 'Detection_Rate', 'False_Activation_Rate', 'Mean_Latency_ms']], 
                                on='Classifier', how='left')
        summary_df.to_csv("Results/combined_performance_summary.csv", index=False)
        print(f"\nCombined performance summary saved to: Results/combined_performance_summary.csv")


    for classifier_name in combined_model_results:
        print(f"\n--- Results for {classifier_name} ---")
        visualize_class_performance(combined_model_results, classifier_name, unique_labels)

    print("\n--- Selecting Best Classifier Based on Movement Metrics ---")
    # FIXED: Pass scalar values instead of dictionaries
    best_classifier_name, best_score, all_scores = select_best_model_for_realtime(
        combined_model_results,
        min_duration=0.5,           # Scalar value
        emg_window_size=window_size, # Scalar value
        confidence_threshold=0.6     # Scalar value
    )
    
    for classifier_name, score in all_scores.items():
        print(f"{classifier_name}: {score:.4f}")

    # Use the selected best classifier
    if best_classifier_name and best_classifier_name in combined_model_results:
        best_pipeline = combined_model_results[best_classifier_name]['pipeline']

        # Save model with metadata
        best_model_data = {
            'pipeline': best_pipeline,
            'window_size': smoothing_window,
            'min_confidence': 0.6,
            'label_names': list(unique_labels),
            'history_size': history_size,
            'composite_score': best_score,
            'movement_metrics': combined_model_results[best_classifier_name].get('realistic_movement_metrics', {}),
            'feature_names': feature_names
        }

        joblib.dump(best_model_data, "Models/best_temporal_emg_model.pkl")
        print(f"Best classifier ({best_classifier_name}) saved as 'best_temporal_emg_model.pkl'")
    else:
        print("No valid best classifier found")

    print("\nEMG classification with temporal context completed successfully.")
if __name__ == "__main__":
    main()