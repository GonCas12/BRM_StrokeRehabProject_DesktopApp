import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
import joblib
from scipy import signal
import threading
import time
import mock_nidaqmx_module as nidaqmx

# Define preprocessing function
def preprocess_emg(emg_data, metadata, apply_filters=True):
    n_channels, n_samples = emg_data.shape
    processed_data = np.zeros_like(emg_data)
    
    # Apply DC offset correction using metadata
    dc_offsets = metadata['dc_offsets']
    sampling_rate = metadata['sampling_rate']
    
    # Design filters if needed
    if apply_filters:
        # Bandpass filter design (20-450Hz)
        lowcut = 20.0
        highcut = 450.0
        nyquist = 0.5 * sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        low = min(0.95, low)
        high = min(0.99, high)
        order_bandpass = 4
        sos_bandpass = signal.butter(order_bandpass, [low, high], btype='band', output='sos')
        
        # Notch filter design (50Hz)
        notch_freq = 50.0
        quality_factor = 30.0
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sampling_rate)
        sos_notch = signal.tf2sos(b_notch, a_notch)
    
    # Process each channel
    for c in range(n_channels):
        channel_data = emg_data[c, :].copy()
        dc_offset = dc_offsets[c] if c < len(dc_offsets) else np.mean(channel_data)
        channel_data = channel_data - dc_offset
        
        if apply_filters:
            channel_data = signal.sosfilt(sos_bandpass, channel_data)
            channel_data = signal.sosfilt(sos_notch, channel_data)
        
        processed_data[c, :] = channel_data
    
    return processed_data

# Exact feature extraction function from your original code
def extract_features_with_context(windows, history_size=3):
    """
    Extract features from segmented EMG data with temporal context - EXACT MATCH to training
    
    Args:
        windows: Segmented EMG data with shape (n_windows, n_channels, window_size)
        history_size: Number of previous windows to consider for context
        
    Returns:
        features: Feature matrix with temporal context
    """
    if windows.shape[0] == 0:
        return np.array([])
    
    n_windows, n_channels, window_size = windows.shape
    
    # Basic features per window (12 features per channel)
    n_features_per_channel = 12
    basic_features = np.zeros((n_windows, n_channels * n_features_per_channel))
    
    # Extract basic features for each window
    for w in range(n_windows):
        feature_idx = 0
        
        for c in range(n_channels):
            signal_data = windows[w, c]
            
            # Basic time domain features
            # 1. Mean Absolute Value (MAV)
            mav = np.mean(np.abs(signal_data))
            
            # 2. Root Mean Square (RMS)
            rms = np.sqrt(np.mean(signal_data**2))
            
            # 3. Zero Crossings (ZC)
            threshold = 0.015 * np.std(signal_data)
            zc = np.sum(np.diff(np.signbit(signal_data)) & (np.abs(np.diff(signal_data)) > threshold))
            
            # 4. Waveform Length (WL)
            wl = np.sum(np.abs(np.diff(signal_data)))
            
            # 5. Signal Complexity (using std)
            entropy = np.std(signal_data)
            
            # 6. Variance
            variance = np.var(signal_data)
            
            # Frequency domain features
            if window_size > 1:
                freqs = np.fft.rfftfreq(window_size, d=1.0/2000)
                ps = np.abs(np.fft.rfft(signal_data))**2
                total_power = np.sum(ps)
                
                # 7. Mean Frequency
                mean_freq = np.sum(freqs * ps) / total_power if total_power > 0 else 0
                
                # 8. Median Frequency
                cum_sum = np.cumsum(ps)
                if total_power > 0:
                    median_idx = np.where(cum_sum >= total_power/2)[0][0] if len(np.where(cum_sum >= total_power/2)[0]) > 0 else 0
                    median_freq = freqs[median_idx]
                else:
                    median_freq = 0
                    
                # 9. Peak Frequency
                peak_freq_idx = np.argmax(ps)
                peak_freq = freqs[peak_freq_idx] if ps[peak_freq_idx] > 0 else 0
                
                # 10-12. Band Powers
                low_band_mask = (freqs >= 20) & (freqs <= 100)
                med_band_mask = (freqs > 100) & (freqs <= 250)
                high_band_mask = (freqs > 250) & (freqs <= 450)
                
                low_band_power = np.sum(ps[low_band_mask]) / total_power if total_power > 0 else 0
                med_band_power = np.sum(ps[med_band_mask]) / total_power if total_power > 0 else 0
                high_band_power = np.sum(ps[high_band_mask]) / total_power if total_power > 0 else 0
            else:
                mean_freq = median_freq = peak_freq = low_band_power = med_band_power = high_band_power = 0
            
            # Store features
            basic_features[w, feature_idx:feature_idx+n_features_per_channel] = [
                mav, rms, zc, wl, entropy, variance,
                mean_freq, median_freq, peak_freq, 
                low_band_power, med_band_power, high_band_power
            ]
            feature_idx += n_features_per_channel
    
    # Now add temporal context (delta features)
    if history_size <= 0 or n_windows <= history_size:
        # If no history requested or not enough windows, return basic features
        return basic_features
    
    # Initialize feature matrix with context
    context_features = np.zeros((n_windows, basic_features.shape[1] * 2))
    
    # First windows without enough history
    context_features[:history_size, :basic_features.shape[1]] = basic_features[:history_size]
    
    # For each window with enough history
    for w in range(history_size, n_windows):
        # Current window features
        context_features[w, :basic_features.shape[1]] = basic_features[w]
        
        # Add the delta (difference from previous)
        history_avg = np.mean(basic_features[w-history_size:w], axis=0)
        context_features[w, basic_features.shape[1]:] = basic_features[w] - history_avg
    
    return context_features

# Final improved classifier class with formatting fix
class FinalEMGClassifier:
    def __init__(self, model_path="best_temporal_emg_model.pkl"):
        print(f"Loading model from {model_path}")
        try:
            model_data = joblib.load(model_path)
            
            # Extract model components
            self.pipeline = model_data['pipeline']
            self.window_size = model_data.get('window_size', 5)
            self.min_confidence = model_data.get('min_confidence', 0.5)
            self.label_names = model_data.get('label_names', ['Rest', 'Flexion', 'Extension'])
            self.history_size = model_data.get('history_size', 3)
            
            print(f"Model loaded successfully with labels: {self.label_names}")
            print(f"Window size: {self.window_size}, Min confidence: {self.min_confidence}")
            
            # Buffer for current window
            self.buffer_size = 500  # Default window size (250ms at 2000Hz)
            self.buffer = None
            self.buffer_filled = False
            self.samples_collected = 0  # Track how many samples we've collected
            
            # Prediction history for smoothing
            self.prediction_history = []
            self.probability_history = []
            
            # State management
            self.current_state = "Rest"
            self.current_confidence = 0.0
            
            # Feature history for context
            self.feature_history = []
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def initialize_buffer(self, n_channels):
        """Initialize buffer with the right number of channels"""
        self.buffer = np.zeros((n_channels, self.buffer_size))
        self.samples_collected = 0
    
    def process_new_data(self, new_data_chunk, metadata):
        """Process new EMG data with improved buffer management"""
        try:
            # Initialize buffer if needed
            if self.buffer is None:
                self.initialize_buffer(new_data_chunk.shape[0])
                print(f"Initialized buffer with shape {self.buffer.shape}")
            
            # Add new data to buffer
            chunk_size = new_data_chunk.shape[1]
            
            if not self.buffer_filled:
                # Add data to buffer
                space_left = self.buffer_size - self.samples_collected
                samples_to_add = min(chunk_size, space_left)
                
                self.buffer[:, self.samples_collected:self.samples_collected+samples_to_add] = new_data_chunk[:, :samples_to_add]
                self.samples_collected += samples_to_add
                
                print(f"Buffer filling: {self.samples_collected}/{self.buffer_size} samples")
                
                # Check if buffer is now full
                if self.samples_collected >= self.buffer_size:
                    self.buffer_filled = True
                    print("Buffer filled completely!")
                else:
                    # Need more data
                    return None, None
            else:
                # Buffer is already filled, update with sliding window
                self.buffer = np.roll(self.buffer, -chunk_size, axis=1)
                self.buffer[:, -chunk_size:] = new_data_chunk
            
            # Now that we have data, process it
            processed_data = preprocess_emg(self.buffer, metadata, apply_filters=True)
            
            # Create window for feature extraction
            window = np.expand_dims(processed_data, axis=0)
            
            # Extract features - using EXACTLY the same function as during training
            #print("Extracting features...")
            basic_features = extract_features_with_context(window, history_size=0)
            
            # Debug output
            #print(f"Extracted {basic_features.shape[1]} features")
            
            # Add to feature history
            self.feature_history.append(basic_features[0])
            if len(self.feature_history) > self.history_size + 1:
                self.feature_history.pop(0)
            
            # Prepare features with context if possible
            if len(self.feature_history) > self.history_size:
                # Current features
                current_features = self.feature_history[-1]
                
                # History average
                history_avg = np.mean(self.feature_history[:-1], axis=0)
                
                # Combine as context features
                context_features = np.concatenate([current_features, current_features - history_avg])
                features = np.expand_dims(context_features, axis=0)
                
                #print(f"Created context features with shape {features.shape}")
            else:
                # Not enough history, use basic features with zeros for context
                # We need to match the expected feature count exactly
                features_with_zeros = np.zeros((1, basic_features.shape[1] * 2))
                features_with_zeros[0, :basic_features.shape[1]] = basic_features[0]
                features = features_with_zeros
                print(f"Not enough history, created padded features with shape {features.shape}")
            
            # Make prediction
            #print("Running classifier pipeline...")
            prediction = self.pipeline.predict(features)[0]
            
            # Get probabilities if available
            confidence = None
            if hasattr(self.pipeline.named_steps['classifier'], "predict_proba"):
                probabilities = self.pipeline.predict_proba(features)[0]
                confidence = np.max(probabilities)
                #print(f"Raw prediction: {prediction}, confidence: {confidence:.4f}")
            else:
                print(f"Raw prediction: {prediction}, no confidence available")
            
            # Apply temporal smoothing
            smoothed_prediction, smoothed_confidence = self._apply_smoothing(prediction, confidence)
            
            # FIXED: The formatting error
            confidence_str = f"{smoothed_confidence:.4f}" if smoothed_confidence is not None else "None"
            #print(f"After smoothing: {smoothed_prediction}, confidence: {confidence_str}")
            
            return smoothed_prediction, smoothed_confidence
            
        except Exception as e:
            print(f"Error in process_new_data: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _apply_smoothing(self, prediction, confidence):
        """Apply temporal smoothing to predictions with safe handling of None values"""
        try:
            # Handle None confidence
            if confidence is None:
                confidence = 0.5  # Default confidence
            
            # Add to history
            self.prediction_history.append(prediction)
            self.probability_history.append(confidence)
            
            # Keep history at desired length
            if len(self.prediction_history) > self.window_size:
                self.prediction_history.pop(0)
                self.probability_history.pop(0)
            
            # Count occurrences of each movement
            movement_counts = {}
            for pred in self.prediction_history:
                movement_counts[pred] = movement_counts.get(pred, 0) + 1
            
            # Find most common movement
            if movement_counts:
                most_common = max(movement_counts.items(), key=lambda x: x[1])[0]
                most_common_count = movement_counts[most_common]
                
                # Calculate average confidence for most common prediction
                avg_confidence = 0.0
                count = 0
                for i, pred in enumerate(self.prediction_history):
                    if pred == most_common:
                        avg_confidence += self.probability_history[i]
                        count += 1
                avg_confidence /= max(1, count)
                
                # Threshold for state change - use lower threshold for testing
                if most_common != self.current_state:
                    # Need high confidence and majority to change state
                    threshold_ratio = 0.6  # Lowered from 0.6 for testing
                    if most_common_count >= len(self.prediction_history) * threshold_ratio and avg_confidence >= self.min_confidence:
                        self.current_state = most_common
                        self.current_confidence = avg_confidence
                else:
                    # Update confidence
                    self.current_confidence = avg_confidence
            
            return self.current_state, self.current_confidence
            
        except Exception as e:
            print(f"Error in smoothing: {e}")
            return None, None


def run_final_test():
    """Run a test of the EMG classifier using the mock DAQ with all fixes applied"""
    print("\n=== Running Final EMG Classifier Test ===\n")
    
    # Find available EMG files
    data_dir = ""
    emg_files = glob.glob(os.path.join(data_dir, "*_emgData_*.npy"))
    
    if not emg_files:
        print(f"No EMG data files found in {data_dir}")
        return
    
    # Group files by movement
    files_by_movement = {}
    for file in emg_files:
        filename = os.path.basename(file)
        parts = filename.split("_")
        
        if len(parts) >= 4:
            movement = parts[3]
            if movement not in files_by_movement:
                files_by_movement[movement] = []
            files_by_movement[movement].append(file)
    
    print(f"Found files for movements: {list(files_by_movement.keys())}")
    
    # Test sequence: Rest -> Flexion -> Extension -> Rest
    test_sequence = []
    for movement in ["Rest", "Flexion", "Extension", "Rest"]:
        if movement in files_by_movement and files_by_movement[movement]:
            test_sequence.append((movement, files_by_movement[movement][0]))
    
    if not test_sequence:
        print("Could not create test sequence. Check your data directory.")
        return
    
    # Initialize classifier
    try:
        classifier = FinalEMGClassifier()
    except Exception as e:
        print(f"Failed to initialize classifier: {e}")
        return
    
    # Function to run test on a single file
    def test_file(movement_name, file_path, max_duration_s=8.0, min_duration_s=0.3):
        """
        Test EMG classifier with rehabilitation-optimized movement detection
        """
        print(f"\n--- Testing {movement_name} data from {os.path.basename(file_path)} ---")
        
        # Movement tracker for rehabilitation context
        class RehabMovementTracker:
            def __init__(self, min_duration_s=0.3, max_duration_s=8.0):
                # Convert to milliseconds
                self.min_duration_ms = min_duration_s * 1000
                self.max_duration_ms = max_duration_s * 1000
                
                self.current_movement = "Rest"
                self.movement_start_time = None
                self.completed_movements = []
                self.movement_detected = False
                self.excessive_duration_warning = False
                
            def update(self, detected_movement, timestamp):
                # Keep track of whether a state change occurs in this update
                state_changed = False
                
                if detected_movement != self.current_movement:
                    # Potential state change
                    if detected_movement != "Rest":
                        # Starting a new movement
                        if self.current_movement == "Rest":
                            # Only reset start time if coming from rest
                            self.movement_start_time = timestamp
                            self.excessive_duration_warning = False
                            state_changed = True
                            print(f"[{timestamp:.2f}s] {detected_movement} movement STARTED")
                        else:
                            # Transitioning from one movement to another without rest
                            print(f"[{timestamp:.2f}s] Movement type changed without rest: {self.current_movement} → {detected_movement}")
                            
                            # Record the previous movement
                            duration = timestamp - self.movement_start_time
                            duration_ms = duration * 1000
                            
                            if duration_ms >= self.min_duration_ms:
                                self.completed_movements.append({
                                    'type': self.current_movement,
                                    'start_time': self.movement_start_time,
                                    'end_time': timestamp,
                                    'duration': duration,
                                    'interrupted': True
                                })
                            
                            # Start tracking new movement
                            self.movement_start_time = timestamp
                            self.excessive_duration_warning = False
                            state_changed = True
                    
                    else:  # Transitioning to Rest
                        # Ending a movement
                        if self.movement_start_time is not None:
                            duration = timestamp - self.movement_start_time
                            duration_ms = duration * 1000
                            
                            if duration_ms >= self.min_duration_ms:
                                # Valid movement
                                self.completed_movements.append({
                                    'type': self.current_movement,
                                    'start_time': self.movement_start_time,
                                    'end_time': timestamp,
                                    'duration': duration,
                                    'interrupted': False
                                })
                                print(f"[{timestamp:.2f}s] {self.current_movement} movement COMPLETED (duration: {duration:.2f}s)")
                            else:
                                # Too short, likely noise
                                print(f"[{timestamp:.2f}s]Ignored short {self.current_movement} (duration: {duration:.2f}s < {min_duration_s}s minimum)")
                        
                        self.movement_start_time = None
                        state_changed = True
                    
                    self.current_movement = detected_movement
                
                # Check for movements that exceed maximum duration
                if self.movement_start_time is not None and self.current_movement != "Rest":
                    duration = timestamp - self.movement_start_time
                    duration_ms = duration * 1000
                    
                    # Only issue the warning once per movement
                    if duration_ms > self.max_duration_ms and not self.excessive_duration_warning:
                        print(f"[{timestamp:.2f}s] {self.current_movement} exceeds recommended duration ({duration:.2f}s > {max_duration_s}s)")
                        # Don't force reset for rehabilitation - patient may need more time
                        self.excessive_duration_warning = True
                
                return state_changed
        
        # Define helper functions for movement detection
        def detect_movement_state(prediction_history, confidence_history, current_movement="Rest"):
            """Apply temporal consensus for rehabilitation context (more forgiving thresholds)"""
            # Count movement types in the prediction history
            movement_counts = {}
            for pred in prediction_history:
                movement_counts[pred] = movement_counts.get(pred, 0) + 1
                
            # Calculate percentages
            total_windows = len(prediction_history)
            movement_percentages = {mov: count/total_windows for mov, count in movement_counts.items()}
            
            # Calculate average confidence for each movement
            movement_confidences = {}
            for mov in movement_counts.keys():
                conf_values = [conf for pred, conf in zip(prediction_history, confidence_history) if pred == mov]
                movement_confidences[mov] = sum(conf_values) / len(conf_values) if conf_values else 0
            
            # REHAB-OPTIMIZED: Much lower thresholds for stroke patients
            onset_percentage_threshold = 0.35  # 35% of windows (was 0.55)
            onset_confidence_threshold = 0.50  # 50% confidence (was 0.65)
            
            # Movement offset threshold (also lower for rehabilitation)
            offset_percentage_threshold = 0.20  # 20% of windows (was 0.25)
            
            # Decision logic
            if current_movement == "Rest":
                # Check for movement onset - prioritize any non-Rest prediction
                for movement in movement_counts.keys():
                    if (movement != "Rest" and
                        movement_percentages[movement] >= onset_percentage_threshold and
                        movement_confidences[movement] >= onset_confidence_threshold):
                        return movement, movement_confidences[movement]
                return "Rest", 1.0
            else:
                # Already in a movement, check for movement offset
                if (current_movement not in movement_percentages or 
                    movement_percentages[current_movement] <= offset_percentage_threshold):
                    return "Rest", 1.0
                return current_movement, movement_confidences[current_movement]
        
        def calculate_signal_energy(emg_data):
            """Calculate normalized energy from EMG data"""
            return np.mean(np.square(emg_data))
        
        # Set the file in the mock DAQ
        nidaqmx.PRE_RECORDED_DATA_FILE = file_path
        nidaqmx.reload_data_file()
        
        # Set up DAQ
        task = nidaqmx.Task()
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0:1", min_val=-5.0, max_val=5.0)
        task.timing.cfg_samp_clk_timing(rate=2000, sample_mode=nidaqmx.AcquisitionType.CONTINUOUS)
        
        # Configure reader
        reader = nidaqmx.stream_readers.AnalogMultiChannelReader(task.in_stream)
        
        # Calibration phase
        print("Starting calibration...")
        calibration_buffer = np.zeros((2, 2 * 2000))  # 2 seconds at 2000Hz
        
        task.start()
        
        # Collect 2 seconds of data for calibration
        samples_collected = 0
        while samples_collected < calibration_buffer.shape[1]:
            samples_to_read = min(200, calibration_buffer.shape[1] - samples_collected)
            temp_buffer = np.zeros((2, samples_to_read))
            reader.read_many_sample(temp_buffer, samples_to_read)
            
            calibration_buffer[:, samples_collected:samples_collected+samples_to_read] = temp_buffer
            samples_collected += samples_to_read
        
        # Calculate DC offset and baseline energy
        dc_offsets = np.mean(calibration_buffer, axis=1)
        baseline_energy = calculate_signal_energy(calibration_buffer)
        print(f"Calibration complete. DC offsets: {dc_offsets}")
        print(f"Baseline energy: {baseline_energy:.6f}")
        
        # Create metadata for classifier
        metadata = {
            'dc_offsets': dc_offsets,
            'sampling_rate': 2000
        }
        
        # Reset classifier's buffer state for clean test
        classifier.buffer_filled = False
        classifier.samples_collected = 0
        classifier.buffer = None
        classifier.feature_history = []
        classifier.prediction_history = []
        classifier.probability_history = []
        classifier.current_state = "Rest"  # Reset to default state
        
        # Initialize tracking variables
        movement_tracker = RehabMovementTracker(min_duration_s=min_duration_s, max_duration_s=max_duration_s)
        prediction_history = []
        confidence_history = []
        energy_history = []
        raw_results = []      # All raw classifier outputs
        validated_results = [] # Final validated movement decisions
        
        # Classification phase
        print("Starting classification...")
        
        # Run classification for 15 seconds (longer for rehabilitation testing)
        start_time = time.time()
        while time.time() - start_time < 15:
            # Read new chunk
            chunk_size = 200  # 100ms at 2000Hz
            temp_buffer = np.zeros((2, chunk_size))
            reader.read_many_sample(temp_buffer, chunk_size)
            
            # Calculate current energy
            current_energy = calculate_signal_energy(temp_buffer)
            energy_ratio = current_energy / baseline_energy
            energy_history.append(current_energy)
            
            # Keep energy history at manageable size
            if len(energy_history) > 30:  # 3 second history
                energy_history.pop(0)
            
            # Process with classifier
            movement, confidence = classifier.process_new_data(temp_buffer, metadata)
            
            if movement is not None:
                # Store current timestamp
                current_time = time.time() - start_time
                
                # Store raw classifier output
                raw_results.append((current_time, movement, confidence, energy_ratio))
                
                # Update prediction history
                prediction_history.append(movement)
                confidence_history.append(confidence)
                
                # Keep history at manageable size (10 windows = 1 second)
                if len(prediction_history) > 10:
                    prediction_history.pop(0)
                    confidence_history.pop(0)
                
                # Apply temporal consensus for stable predictions
                if len(prediction_history) >= 5:  # Need minimum history
                    # Get consensus movement - SIMPLIFIED FOR REHAB
                    consensus_movement, consensus_confidence = detect_movement_state(
                        prediction_history, confidence_history, movement_tracker.current_movement)
                    
                    # CRITICAL FIX: For rehab, we trust the classifier more than energy validation
                    # This is because stroke patients may have weak signals
                    final_movement = consensus_movement
                    final_confidence = consensus_confidence
                    
                    # Only use energy to detect obvious noise (4x above baseline)
                    if energy_ratio > 4.0 and final_movement == "Rest":
                        print(f"[{current_time:.2f}s] High energy detected during Rest state ({energy_ratio:.2f}x)")
                    
                    # Update movement tracker and check if state changed
                    state_changed = movement_tracker.update(final_movement, current_time)
                    
                    # Store validated result
                    validated_results.append((current_time, final_movement, final_confidence, state_changed))
                    
                    # Print real-time update only on state changes or periodically
                    if state_changed or len(validated_results) % 20 == 0:
                        print(f"[{current_time:.2f}s] Current state: {final_movement} (conf: {final_confidence:.2f}, energy: {energy_ratio:.2f}x)")
        
        # Cleanup
        task.stop()
        task.close()
        
        # Ensure last movement is properly closed if still active
        if movement_tracker.current_movement != "Rest" and movement_tracker.movement_start_time is not None:
            duration = (time.time() - start_time) - movement_tracker.movement_start_time
            if duration >= min_duration_s:
                movement_tracker.completed_movements.append({
                    'type': movement_tracker.current_movement,
                    'start_time': movement_tracker.movement_start_time,
                    'end_time': time.time() - start_time,
                    'duration': duration,
                    'interrupted': True  # Marked as interrupted since test ended during movement
                })
        
        # =============== MOVEMENT ANALYSIS AND REPORTING ===============
        
        print("\n=============== MOVEMENT ANALYSIS REPORT ===============")
        print(f"Test file: {os.path.basename(file_path)} (Expected: {movement_name})")
        print(f"Test duration: {time.time() - start_time:.1f} seconds")
        
        # 1. Count raw predictions (classifier output before temporal smoothing)
        raw_movement_counts = {}
        for _, movement, _, _ in raw_results:
            raw_movement_counts[movement] = raw_movement_counts.get(movement, 0) + 1
        
        total_raw = len(raw_results)
        print("\n--- Raw Classifier Output (Before Smoothing) ---")
        for movement, count in raw_movement_counts.items():
            percentage = (count / total_raw) * 100
            print(f"  {movement}: {count}/{total_raw} ({percentage:.1f}%)")
        
        # 2. Count validated predictions (after temporal consensus)
        validated_movement_counts = {}
        for _, movement, _, _ in validated_results:
            validated_movement_counts[movement] = validated_movement_counts.get(movement, 0) + 1
        
        total_validated = len(validated_results)
        print("\n--- Validated Predictions (After Smoothing) ---")
        for movement, count in validated_movement_counts.items():
            percentage = (count / total_validated) * 100
            print(f"  {movement}: {count}/{total_validated} ({percentage:.1f}%)")
        
        # 3. Completed movement summary
        print(f"\n--- Completed Movements ({len(movement_tracker.completed_movements)}) ---")
        if movement_tracker.completed_movements:
            movement_durations = {}
            for i, mov in enumerate(movement_tracker.completed_movements):
                # Build a summary string with emojis for visibility
                status = "INTERRUPTED" if mov['interrupted'] else "COMPLETED"
                print(f"{i+1}. {mov['type']}: {mov['start_time']:.2f}s - {mov['end_time']:.2f}s "
                    f"(duration: {mov['duration']:.2f}s) - {status}")
                
                # Collect duration stats by movement type
                if mov['type'] not in movement_durations:
                    movement_durations[mov['type']] = []
                movement_durations[mov['type']].append(mov['duration'])
            
            # Print duration statistics
            print("\n--- Movement Duration Statistics ---")
            for movement, durations in movement_durations.items():
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
                print(f"  {movement}: {len(durations)} movements, "
                    f"Avg: {avg_duration:.2f}s, Min: {min_duration:.2f}s, Max: {max_duration:.2f}s")
        else:
            print("  No completed movements detected")
        
        # 4. Target accuracy report - how well movements matched the expected type
        if movement_tracker.completed_movements:
            print(f"\nTARGET ACCURACY REPORT FOR {movement_name}")
            
            # Count movements by type
            movement_type_counts = {}
            for mov in movement_tracker.completed_movements:
                movement_type_counts[mov['type']] = movement_type_counts.get(mov['type'], 0) + 1
            
            total_movements = len(movement_tracker.completed_movements)
            
            # Count on-target and off-target movements
            on_target_count = movement_type_counts.get(movement_name, 0)
            off_target_count = total_movements - on_target_count
            
            on_target_percent = (on_target_count / total_movements) * 100 if total_movements > 0 else 0
            off_target_percent = (off_target_count / total_movements) * 100 if total_movements > 0 else 0
            
            print(f"  On Target:  {on_target_count}/{total_movements} ({on_target_percent:.1f}%)")
            print(f"  Off Target: {off_target_count}/{total_movements} ({off_target_percent:.1f}%)")
            
            # If off target, show what it was misclassified as
            if off_target_count > 0:
                print("\n  Misclassifications:")
                for mov_type, count in movement_type_counts.items():
                    if mov_type != movement_name:
                        misclass_percent = (count / off_target_count) * 100
                        print(f"    {mov_type}: {count} ({misclass_percent:.1f}%)")
        
        # 5. Rehabilitation-specific metrics
        print("\n--- Rehabilitation Assessment ---")
        
        if movement_name != "Rest" and movement_tracker.completed_movements:
            # Calculate relevant metrics for rehab progress
            target_movements = [m for m in movement_tracker.completed_movements if m['type'] == movement_name]
            
            if target_movements:
                # Movement frequency
                movements_per_minute = len(target_movements) / ((time.time() - start_time) / 60)
                
                # Average duration
                avg_duration = sum(m['duration'] for m in target_movements) / len(target_movements)
                
                # Duration consistency (coefficient of variation)
                durations = [m['duration'] for m in target_movements]
                duration_std = np.std(durations)
                duration_cv = (duration_std / avg_duration) * 100 if avg_duration > 0 else 0
                
                # Movement quality assessment
                interrupted_count = sum(1 for m in target_movements if m['interrupted'])
                completion_rate = ((len(target_movements) - interrupted_count) / len(target_movements)) * 100
                
                print(f"Movement frequency: {movements_per_minute:.2f} {movement_name}/minute")
                print(f"Average duration: {avg_duration:.2f} seconds")
                print(f"Duration consistency: {duration_cv:.1f}% variation (lower is better)")
                print(f"Completion rate: {completion_rate:.1f}% (% of movements properly completed)")
                
                # Guidance based on results
                print("\n  Rehabilitation Guidance:")
                if avg_duration > max_duration_s:
                    print(f" Movements are taking longer than the target ({max_duration_s}s). Focus on increasing speed.")
                elif avg_duration < min_duration_s * 1.5:
                    print(f"Movements are very quick. Consider focusing on control and form.")
                
                if duration_cv > 30:
                    print(f"Movement duration is inconsistent. Work on maintaining steady rhythm.")
                
                if completion_rate < 80:
                    print(f"Many movements were interrupted. Focus on completing full range of motion.")
                
                if movements_per_minute < 2 and movement_name != "Rest":
                    print(f"Movement frequency is low. Try to increase number of repetitions.")
            else:
                print(f"  No {movement_name} movements detected for rehabilitation assessment.")
        elif movement_name == "Rest":
            rest_percentage = validated_movement_counts.get("Rest", 0) / total_validated * 100
            print(f"  Rest maintenance: {rest_percentage:.1f}% (higher is better for Rest exercises)")
            
            if rest_percentage < 70:
                print(" Difficulty maintaining rest state. Focus on relaxation techniques.")
        
        # Add a note about the raw vs validated predictions
        if "Rest" in validated_movement_counts and validated_movement_counts["Rest"] == total_validated:
            if any(mov != "Rest" for mov, _, _, _ in raw_results):
                print("\n NOTE: Raw classifier detected movements, but they didn't meet the")
                print("   consistency threshold. Consider decreasing onset thresholds for")
                print("   rehabilitation settings or checking electrode placement.")
        
        print("\n========================================================")
        
        return movement_tracker.completed_movements
    
    # Run test for each file in sequence
    for movement, file_path in test_sequence:
        test_file(movement, file_path)
        
        # Short pause between files
        time.sleep(1)
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    run_final_test()