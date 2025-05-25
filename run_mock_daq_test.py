import numpy as np
import matplotlib.pyplot as plt
import time
import os
import glob
import joblib
from scipy import signal
import threading
import zmq
import json
import argparse
import sys
import mock_nidaqmx_module as nidaqmx  # Replace with real nidaqmx for production

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

class EMGClassifier:
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
            
            # Movement detection
            self.movement_detector = MovementDetector()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def initialize_buffer(self, n_channels):
        """Initialize buffer with the right number of channels"""
        self.buffer = np.zeros((n_channels, self.buffer_size))
        self.samples_collected = 0
    
    def process_new_data(self, new_data_chunk, metadata):
        """Process new EMG data"""
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
                
                if self.samples_collected >= self.buffer_size:
                    self.buffer_filled = True
                else:
                    # Need more data
                    return None, None, None
            else:
                # Buffer is already filled, update with sliding window
                self.buffer = np.roll(self.buffer, -chunk_size, axis=1)
                self.buffer[:, -chunk_size:] = new_data_chunk
            
            # Now that we have data, process it
            processed_data = preprocess_emg(self.buffer, metadata, apply_filters=True)
            
            # Calculate signal intensity (RMS of processed data)
            signal_intensity = np.sqrt(np.mean(processed_data**2))
            
            # Create window for feature extraction
            window = np.expand_dims(processed_data, axis=0)
            
            # Extract features - using EXACTLY the same function as during training
            basic_features = extract_features_with_context(window, history_size=0)
            
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
            else:
                # Not enough history, use basic features with zeros for context
                features_with_zeros = np.zeros((1, basic_features.shape[1] * 2))
                features_with_zeros[0, :basic_features.shape[1]] = basic_features[0]
                features = features_with_zeros
            
            # Make prediction
            prediction = self.pipeline.predict(features)[0]
            
            # Get probabilities if available
            confidence = None
            if hasattr(self.pipeline.named_steps['classifier'], "predict_proba"):
                probabilities = self.pipeline.predict_proba(features)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 0.7  # Default confidence if not available
            
            # Apply temporal smoothing
            smoothed_prediction, smoothed_confidence = self._apply_smoothing(prediction, confidence)
            
            # Update movement detector
            self.movement_detector.update(smoothed_prediction, smoothed_confidence, signal_intensity)
            
            return smoothed_prediction, smoothed_confidence, signal_intensity
            
        except Exception as e:
            print(f"Error in process_new_data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
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
                
                # Threshold for state change - lower for stroke rehabilitation
                if most_common != self.current_state:
                    # Need high confidence and majority to change state
                    threshold_ratio = 0.3  # Lower for stroke patients
                    if most_common_count >= len(self.prediction_history) * threshold_ratio and avg_confidence >= self.min_confidence * 0.8:
                        self.current_state = most_common
                        self.current_confidence = avg_confidence
                else:
                    # Update confidence
                    self.current_confidence = avg_confidence
            
            return self.current_state, self.current_confidence
            
        except Exception as e:
            print(f"Error in smoothing: {e}")
            return None, None

class MovementDetector:
    """Detects and tracks discrete movements"""
    def __init__(self, min_duration_s=0.3):
        self.min_duration_ms = min_duration_s * 1000
        self.current_movement = "Rest"
        self.movement_start_time = None
        self.completed_movements = []
        self.last_update_time = time.time()
    
    def update(self, movement, confidence, intensity):
        """Update the movement detector with new classification"""
        current_time = time.time()
        time_delta = current_time - self.last_update_time
        self.last_update_time = current_time
        
        state_changed = False
        
        if movement != self.current_movement:
            # Potential state change
            if movement != "Rest":
                # Starting a new movement
                if self.current_movement == "Rest":
                    self.movement_start_time = current_time
                    state_changed = True
                else:
                    # Transitioning from one movement to another without rest
                    # Record the previous movement
                    if self.movement_start_time is not None:
                        duration = current_time - self.movement_start_time
                        duration_ms = duration * 1000
                        
                        if duration_ms >= self.min_duration_ms:
                            self.completed_movements.append({
                                'type': self.current_movement,
                                'start_time': self.movement_start_time,
                                'end_time': current_time,
                                'duration': duration
                            })
                    
                    # Start tracking new movement
                    self.movement_start_time = current_time
                    state_changed = True
            else:  # Transitioning to Rest
                # Ending a movement
                if self.movement_start_time is not None:
                    duration = current_time - self.movement_start_time
                    duration_ms = duration * 1000
                    
                    if duration_ms >= self.min_duration_ms:
                        # Valid movement
                        self.completed_movements.append({
                            'type': self.current_movement,
                            'start_time': self.movement_start_time,
                            'end_time': current_time,
                            'duration': duration
                        })
                
                self.movement_start_time = None
                state_changed = True
            
            self.current_movement = movement
        
        return state_changed

class EMGZmqBridge:
    """Bridge between EMG classifier and ZMQ publisher"""
    def __init__(self, port=5555, model_path="best_temporal_emg_model.pkl"):
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{port}")
        print(f"ZMQ publisher started on port {port}")
        
        # EMG classifier setup
        self.classifier = EMGClassifier(model_path)
        
        # Buffer for plot data
        self.plot_buffer_size = 200
        self.plot_buffer = np.zeros(self.plot_buffer_size)
        
        # State
        self.running = False
        self.thread = None
    
    def start(self, use_mock=True, mock_file=None):
        """Start the bridge"""
        self.running = True
        self.thread = threading.Thread(target=self._run_bridge, args=(use_mock, mock_file))
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the bridge"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.socket.close()
        self.context.term()
        print("EMG ZMQ bridge stopped")
    
    def _run_bridge(self, use_mock=True, mock_file=None):
        """Run the bridge in a loop"""
        try:
            # Set up DAQ
            if use_mock:
                if mock_file:
                    # Use specific mock file
                    nidaqmx.PRE_RECORDED_DATA_FILE = mock_file
                    nidaqmx.reload_data_file()
                    print(f"Using mock DAQ with pre-recorded data: {mock_file}")
                else:
                    print("Using mock DAQ with simulated data")
            else:
                print("Using real DAQ")
            
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
            baseline_energy = np.mean(np.square(calibration_buffer))
            print(f"Calibration complete. DC offsets: {dc_offsets}")
            self.dc_offsets = dc_offsets
            
            # Create metadata for classifier
            metadata = {
                'dc_offsets': dc_offsets,
                'sampling_rate': 2000
            }
            
            # Main processing loop
            msg_count = 0
            last_time = time.time()
            
            while self.running:
                # Read new chunk
                chunk_size = 200  # 200ms at 2000Hz
                temp_buffer = np.zeros((2, chunk_size))
                reader.read_many_sample(temp_buffer, chunk_size)
                
                # Process with classifier
                movement, confidence, intensity = self.classifier.process_new_data(temp_buffer, metadata)
                
                if movement is not None:
                    # Update plot buffer
                    # Update plot buffer - get mean of absolute values across channels
                    signal = np.mean(np.abs(temp_buffer), axis=0)

                    # Apply DC offset correction to each channel separately
                    temp_buffer_corrected = temp_buffer.copy()
                    for i in range(2):  # For each channel
                        temp_buffer_corrected[i, :] = temp_buffer[i, :] - self.dc_offsets[i]

                    # Update the plot buffer
                    self.plot_buffer = np.roll(self.plot_buffer, -len(signal))
                    self.plot_buffer[-len(signal):] = signal

                    # Scale to microvolts for better visualization
                    plot_data = temp_buffer_corrected * 1000  # Convert to μV
                    
                    # Calculate status for app
                    if movement == "Rest":
                        status = "idle"
                    else:
                        status = "active"
                    
                    # Create message
                    msg_count += 1
                    message = {
                        'timestamp': time.time(),
                        'status': status,
                        'movement': movement,
                        'confidence': float(confidence) if confidence is not None else 0.0,
                        'intensity': float(intensity) if intensity is not None else 0.0,
                        'plot_data': plot_data.tolist()
                    }
                    
                    # Send over ZMQ
                    self.socket.send_string(json.dumps(message))
                    
                    # Log periodically
                    if msg_count % 100 == 0:
                        now = time.time()
                        rate = 100 / (now - last_time)
                        print(f"Sent {msg_count} messages. Rate: {rate:.1f} msg/sec. Last movement: {movement} ({confidence:.2f})")
                        last_time = now
                
                # Small delay to prevent CPU hammering
                time.sleep(0.01)
            
            # Cleanup
            task.stop()
            task.close()
            print("DAQ task stopped")
            
        except Exception as e:
            print(f"Error in bridge: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='EMG ZMQ Bridge')
    parser.add_argument('--port', type=int, default=5555, help='ZMQ port to bind to')
    parser.add_argument('--model', type=str, default='best_temporal_emg_model.pkl', help='Path to the model file')
    parser.add_argument('--mock', action='store_true', help='Use mock DAQ instead of real DAQ')
    parser.add_argument('--file', type=str, help='Pre-recorded data file for mock DAQ')
    
    args = parser.parse_args()
    
    print("Starting EMG ZMQ Bridge...")
    print(f"Port: {args.port}")
    print(f"Model: {args.model}")
    print(f"Mock: {args.mock}")
    print(f"File: {args.file}")
    
    bridge = EMGZmqBridge(port=args.port, model_path=args.model)
    bridge.start(use_mock=args.mock, mock_file=args.file)
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping bridge...")
        bridge.stop()
        print("Bridge stopped")

if __name__ == "__main__":
    main()