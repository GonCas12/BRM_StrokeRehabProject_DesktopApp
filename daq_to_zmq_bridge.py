#!/usr/bin/env python3
"""
DAQ-to-ZMQ Bridge - Reads EMG data from DAQ and sends it to BRM_StrokeRehab application
with advanced temporal EMG classifier integration
"""

import time
import numpy as np
import scipy.signal
import threading
import zmq
import json
import argparse
import os
from enum import Enum
import joblib

# Conditional import for nidaqmx
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    from nidaqmx import errors
    import nidaqmx.stream_readers
    system = nidaqmx.system.System.local()
    print(f"NI-DAQmx system found. Driver version: {system.driver_version}")
    if not system.devices:
        print("WARNING: No NI DAQ devices detected!")
    NIDAQMX_AVAILABLE = True
except ImportError:
    print("WARNING: nidaqmx library not found. Running in simulation mode.")
    NIDAQMX_AVAILABLE = False

# EMG Status Enum
class EMGStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"

class RealTimeTemporalEMGClassifier:
    """
    Real-time EMG classifier with temporal smoothing and history
    """
    def __init__(self, model_path="best_temporal_emg_model.pkl"):
        try:
            # Load model
            print(f"Loading EMG classifier model from {model_path}")
            model_data = joblib.load(model_path)
            
            # Extract model components
            self.pipeline = model_data['pipeline']
            self.window_size = model_data.get('window_size', 5)
            self.min_confidence = model_data.get('min_confidence', 0.6)
            self.label_names = model_data.get('label_names', ['Rest', 'Flexion', 'Extension'])
            self.history_size = model_data.get('history_size', 3)
            
            # Buffer for current window
            self.buffer_size = 500  # Default window size (250ms at 2000Hz)
            self.buffer = None
            self.buffer_filled = False
            
            # Prediction history for smoothing
            self.prediction_history = []
            self.probability_history = []
            
            # State management
            self.current_state = "Rest"
            self.current_confidence = 0.0
            
            # Feature history for context
            self.feature_history = []
            print(f"Classifier initialized with {len(self.label_names)} movement classes: {self.label_names}")

        except FileNotFoundError:
            print(f"ERROR: Model file {model_path} not found!")
            raise
        except Exception as e:
            print(f"ERROR: Failed to load classifier model: {e}")
            raise
    
    def initialize_buffer(self, n_channels):
        """Initialize buffer with the right number of channels"""
        self.buffer = np.zeros((n_channels, self.buffer_size))
    
    def process_new_data(self, new_data, metadata):
        """
        Process a new chunk of EMG data
        
        Args:
            new_data: EMG data with shape (n_channels, n_samples)
            metadata: Dictionary with metadata information
            
        Returns:
            movement: Detected movement
            confidence: Confidence in the detection
        """
        # Initialize buffer if needed
        if self.buffer is None:
            self.initialize_buffer(new_data.shape[0])
        
        # Add new data to buffer (sliding window)
        if new_data.shape[1] >= self.buffer_size:
            # If new data is larger than buffer, just take the most recent buffer_size samples
            self.buffer = new_data[:, -self.buffer_size:]
            self.buffer_filled = True
        else:
            # Update buffer with new data
            self.buffer = np.roll(self.buffer, -new_data.shape[1], axis=1)
            self.buffer[:, -new_data.shape[1]:] = new_data
            self.buffer_filled = True
        
        # Process current buffer
        if not self.buffer_filled:
            return None, None
        
        return self._process_buffer(metadata)
    
    def _process_buffer(self, metadata):
        """Process current buffer and make prediction with temporal context"""
        # Preprocess buffer
        processed_data = self._preprocess_emg(self.buffer, metadata)
        
        # Create single window for feature extraction
        window = np.expand_dims(processed_data, axis=0)  # Shape: (1, n_channels, window_size)
        
        # Extract basic features (without context)
        basic_features = self._extract_features(window)
        
        # Add to feature history
        self.feature_history.append(basic_features[0])
        if len(self.feature_history) > self.history_size + 1:
            self.feature_history.pop(0)
        
        # If we have enough history, create context features
        if len(self.feature_history) > self.history_size:
            # Current features
            current_features = self.feature_history[-1]
            
            # History average
            history_avg = np.mean(self.feature_history[:-1], axis=0)
            
            # Combine as context features
            context_features = np.concatenate([current_features, current_features - history_avg])
            features = np.expand_dims(context_features, axis=0)
        else:
            # Not enough history, pad with zeros
            n_features = basic_features.shape[1]
            features = np.pad(
                basic_features, 
                ((0, 0), (0, n_features)),
                'constant'
            )
        
        # Make prediction
        prediction = self.pipeline.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(self.pipeline.named_steps['classifier'], "predict_proba"):
            probabilities = self.pipeline.predict_proba(features)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0
        
        # Apply temporal smoothing
        smoothed_prediction, smoothed_confidence = self._apply_smoothing(prediction, confidence)
        
        return smoothed_prediction, smoothed_confidence
    
    def _preprocess_emg(self, emg_data, metadata):
        """Preprocess EMG data using metadata for offset correction"""
        n_channels, n_samples = emg_data.shape
        processed_data = np.zeros_like(emg_data)
        
        # Get DC offsets and sampling rate from metadata
        dc_offsets = metadata['dc_offsets']
        sampling_rate = metadata['sampling_rate']
        
        # Process each channel
        for c in range(n_channels):
            # Get channel data
            channel_data = emg_data[c, :].copy()
            
            # Apply DC offset correction
            dc_offset = dc_offsets[c] if c < len(dc_offsets) else np.mean(channel_data)
            channel_data = channel_data - dc_offset
            
            # Store processed data
            processed_data[c, :] = channel_data
        
        return processed_data
    
    def _extract_features(self, windows):
        """Extract basic features from EMG data"""
        if windows.shape[0] == 0:
            return np.array([])
        
        n_windows, n_channels, window_size = windows.shape
        
        # Define number of features per channel
        n_features_per_channel = 12
        features = np.zeros((n_windows, n_channels * n_features_per_channel))
        
        # Extract features for each window
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
                features[w, feature_idx:feature_idx+n_features_per_channel] = [
                    mav, rms, zc, wl, entropy, variance,
                    mean_freq, median_freq, peak_freq, 
                    low_band_power, med_band_power, high_band_power
                ]
                feature_idx += n_features_per_channel
        
        return features
    
    def _apply_smoothing(self, prediction, confidence):
        """Apply temporal smoothing to predictions"""
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
            
            # Threshold for state change
            if most_common != self.current_state:
                # Need high confidence and majority to change state
                threshold_ratio = 0.6  # At least 60% of windows
                if most_common_count >= len(self.prediction_history) * threshold_ratio and avg_confidence >= self.min_confidence:
                    self.current_state = most_common
                    self.current_confidence = avg_confidence
            else:
                # Update confidence
                self.current_confidence = avg_confidence
        
        return self.current_state, self.current_confidence

# Configuration
class Config:
    # DAQ Configuration
    USE_SIMULATION = not NIDAQMX_AVAILABLE  # Auto-detect, but can be overridden
    DAQ_DEVICE_NAME = "Dev1"
    DAQ_CHANNELS_STR = "ai0:3"  # Will use the first 4 analog inputs
    DAQ_SAMPLING_RATE_HZ = 1000
    DAQ_VOLTAGE_RANGE_MIN = -5.0
    DAQ_VOLTAGE_RANGE_MAX = 5.0
    TERMINAL_CONFIG = TerminalConfiguration.DEFAULT if NIDAQMX_AVAILABLE else None
    
    # Processing Configuration
    CHUNK_SIZE = 100  # Process 100 samples at a time
    CHUNK_DURATION_MS = CHUNK_SIZE * 1000 // DAQ_SAMPLING_RATE_HZ
    INTENSITY_THRESHOLD = 0.2  # Threshold to determine active vs idle
    APPLY_FILTERS = True
    
    # ZMQ Configuration
    ZMQ_PUBLISHER_PORT = 5555
    ZMQ_PROTOCOL = "tcp"
    ZMQ_BIND_ADDRESS = "*"  # Bind to all interfaces
    
    # EMG Feature extraction
    RMS_WINDOW_SIZE = 50  # Window size for RMS calculation (in samples)
    
    # EMG Classifier Configuration
    CLASSIFIER_MODEL_PATH = "best_temporal_emg_model.pkl"  # Path to your trained model
    USE_ADVANCED_CLASSIFIER = True  # Set to False to revert to simple threshold
    MOVEMENT_IDLE_THRESHOLD = 0.5  # Confidence threshold
    
    # Buffer for temporal features
    BUFFER_SIZE = 500  # Size of buffer in samples (250ms at 2000Hz) - must match model
    MOVEMENT_LABELS = ["Rest", "Flexion", "Extension"]  # Movement classes in your model
    
    # Convert channels string to number of channels
    @classmethod
    def get_num_channels(cls):
        # Parse channel range like "ai0:3" to get 4 channels
        try:
            if ":" in cls.DAQ_CHANNELS_STR:
                parts = cls.DAQ_CHANNELS_STR.split(":")
                start = int(parts[0].replace("ai", ""))
                end = int(parts[1])
                return end - start + 1
            else:
                return 1  # Single channel
        except:
            return 4  # Default to 4 channels


class DAQtoZMQBridge:
    def __init__(self, config=None):
        self.config = config or Config()
        self.running = False
        self.daq_task = None
        self.daq_reader = None
        self.num_channels = self.config.get_num_channels()
        print(f"Initialized with {self.num_channels} channels")
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.bind_address = f"{self.config.ZMQ_PROTOCOL}://{self.config.ZMQ_BIND_ADDRESS}:{self.config.ZMQ_PUBLISHER_PORT}"
        self.socket.bind(self.bind_address)
        print(f"ZMQ publisher bound to {self.bind_address}")
        
        # Filter design
        self.filter_sos_bandpass = None
        self.filter_sos_notch = None
        self.filter_zi = {}
        
        if self.config.APPLY_FILTERS:
            self._design_filters()
            
        # Initialize buffer for classifier
        self.emg_buffer = np.zeros((self.num_channels, self.config.BUFFER_SIZE))
        self.buffer_position = 0
        self.buffer_filled = False
        
        # Initialize our advanced classifier
        if self.config.USE_ADVANCED_CLASSIFIER:
            try:
                if os.path.exists(self.config.CLASSIFIER_MODEL_PATH):
                    self.classifier = RealTimeTemporalEMGClassifier(self.config.CLASSIFIER_MODEL_PATH)
                    print(f"Loaded temporal EMG classifier from {self.config.CLASSIFIER_MODEL_PATH}")
                else:
                    print(f"Warning: Model file {self.config.CLASSIFIER_MODEL_PATH} not found. Using simple threshold.")
                    self.classifier = None
                    self.config.USE_ADVANCED_CLASSIFIER = False
            except Exception as e:
                print(f"Error loading classifier: {e}")
                self.classifier = None
                self.config.USE_ADVANCED_CLASSIFIER = False
        else:
            self.classifier = None
        
        # Prediction state
        self.current_movement = "Rest"
        self.current_confidence = 0.0
        
        # Add metadata for classifier
        self.metadata = {
            'dc_offsets': np.zeros(self.num_channels),  # Will be updated during calibration
            'sampling_rate': self.config.DAQ_SAMPLING_RATE_HZ
        }
    
    def _design_filters(self):
        """Design the digital filters for EMG processing"""
        try:
            # Band-Pass Filter (20-450 Hz for EMG)
            lowcut = 20.0  # Hz
            highcut = 450.0  # Hz
            nyquist = 0.5 * self.config.DAQ_SAMPLING_RATE_HZ
            low = lowcut / nyquist
            high = highcut / nyquist
            order_bandpass = 4
            self.filter_sos_bandpass = scipy.signal.butter(
                order_bandpass, [low, high], btype='band', output='sos'
            )
            print(f"Designed Band-pass filter: {lowcut}-{highcut} Hz, Order {order_bandpass}")
            
            # Notch Filter (50/60 Hz for power line interference)
            notch_freq = 50.0  # Hz (or 60 Hz in some regions)
            quality_factor = 30.0
            b_notch, a_notch = scipy.signal.iirnotch(
                notch_freq, quality_factor, self.config.DAQ_SAMPLING_RATE_HZ
            )
            self.filter_sos_notch = scipy.signal.tf2sos(b_notch, a_notch)
            print(f"Designed Notch filter: {notch_freq} Hz")
            
            # Initialize filter states
            for i in range(self.num_channels):
                self.filter_zi[f'bandpass_ch{i}'] = scipy.signal.sosfilt_zi(self.filter_sos_bandpass)
                self.filter_zi[f'notch_ch{i}'] = scipy.signal.sosfilt_zi(self.filter_sos_notch)
                
        except Exception as e:
            print(f"Error designing filters: {e}")
            self.filter_sos_bandpass = None
            self.filter_sos_notch = None
    
    def _apply_filters(self, data):
        """Apply the designed filters to the EMG data"""
        if not self.config.APPLY_FILTERS:
            return data
            
        filtered_data = data.copy()
        
        for i in range(self.num_channels):
            channel_data = filtered_data[i, :]
            
            # Apply Band-pass filter
            if self.filter_sos_bandpass is not None:
                channel_data, self.filter_zi[f'bandpass_ch{i}'] = scipy.signal.sosfilt(
                    self.filter_sos_bandpass, channel_data, zi=self.filter_zi[f'bandpass_ch{i}']
                )
                
            # Apply Notch filter
            if self.filter_sos_notch is not None:
                channel_data, self.filter_zi[f'notch_ch{i}'] = scipy.signal.sosfilt(
                    self.filter_sos_notch, channel_data, zi=self.filter_zi[f'notch_ch{i}']
                )
                
            filtered_data[i, :] = channel_data
            
        return filtered_data
    
    def _update_buffer(self, new_data):
        """Update the EMG buffer with new data"""
        chunk_size = new_data.shape[1]
        
        # If new data is larger than buffer, just take the most recent buffer_size samples
        if chunk_size >= self.config.BUFFER_SIZE:
            self.emg_buffer = new_data[:, -self.config.BUFFER_SIZE:]
            self.buffer_filled = True
            return
        
        # If buffer isn't filled yet
        if not self.buffer_filled:
            # Check if this chunk will fill the buffer
            if self.buffer_position + chunk_size >= self.config.BUFFER_SIZE:
                # This chunk will complete the buffer
                space_remaining = self.config.BUFFER_SIZE - self.buffer_position
                self.emg_buffer[:, self.buffer_position:] = new_data[:, :space_remaining]
                self.buffer_filled = True
                # If there's extra data, roll buffer and add remaining data
                if chunk_size > space_remaining:
                    extra_data = chunk_size - space_remaining
                    self.emg_buffer = np.roll(self.emg_buffer, -extra_data, axis=1)
                    self.emg_buffer[:, -extra_data:] = new_data[:, space_remaining:]
            else:
                # Won't fill buffer yet, just add the data
                self.emg_buffer[:, self.buffer_position:self.buffer_position+chunk_size] = new_data
                self.buffer_position += chunk_size
        else:
            # Buffer is already filled, use sliding window approach
            self.emg_buffer = np.roll(self.emg_buffer, -chunk_size, axis=1)
            self.emg_buffer[:, -chunk_size:] = new_data
    
    def _determine_status(self, intensity, movement, confidence):
        """
        Determine EMG status based on movement prediction and confidence
        
        Args:
            intensity: Overall signal intensity (0-1)
            movement: Predicted movement type ("Rest", "Flexion", "Extension")
            confidence: Classification confidence (0-1)
        
        Returns:
            EMGStatus: IDLE or ACTIVE
        """
        # Default to IDLE
        status = EMGStatus.IDLE
        
        # First check if we're using the classifier
        if self.config.USE_ADVANCED_CLASSIFIER and self.classifier:
            # Movement is active if:
            # 1. Not rest AND
            # 2. Confidence is above threshold
            if movement != "Rest" and confidence >= self.config.MOVEMENT_IDLE_THRESHOLD:
                status = EMGStatus.ACTIVE
        else:
            # Fallback to simple threshold
            if intensity > self.config.INTENSITY_THRESHOLD:
                status = EMGStatus.ACTIVE
        
        return status

    def _calculate_features_and_predict(self, filtered_data):
        """
        Calculate features and predict movement using temporal classifier
        
        Returns:
            tuple: (intensity, movement_type, confidence)
        """
        # Calculate basic intensity (for backward compatibility)
        rms_values = np.zeros(self.num_channels)
        for i in range(self.num_channels):
            rms_values[i] = np.sqrt(np.mean(np.square(filtered_data[i, :])))
        intensity = np.mean(rms_values)
        normalized_intensity = np.clip(intensity / 1.0, 0.0, 1.0)
        
        # Default return values
        movement = "Rest"
        confidence = 0.0
        
        # If buffer is filled and we have a classifier, make prediction
        if self.buffer_filled and self.classifier and self.config.USE_ADVANCED_CLASSIFIER:
            # Get prediction from advanced classifier
            prediction_result = self.classifier.process_new_data(self.emg_buffer, self.metadata)
            
            if prediction_result is not None:
                movement, confidence = prediction_result
                
                # Update intensity based on confidence and movement type
                if movement != "Rest":
                    # Scale intensity with confidence for non-rest movements
                    normalized_intensity = max(normalized_intensity, confidence * 0.8)
        
        return normalized_intensity, movement, confidence
    
    def _init_daq(self):
        """Initialize the DAQ task for EMG acquisition"""
        if self.config.USE_SIMULATION:
            print("Using simulated DAQ")
            return True
            
        try:
            self.daq_task = nidaqmx.Task()
            self.daq_task.ai_channels.add_ai_voltage_chan(
                f"{self.config.DAQ_DEVICE_NAME}/{self.config.DAQ_CHANNELS_STR}",
                terminal_config=self.config.TERMINAL_CONFIG,
                min_val=self.config.DAQ_VOLTAGE_RANGE_MIN,
                max_val=self.config.DAQ_VOLTAGE_RANGE_MAX
            )
            
            buffer_size = self.config.DAQ_SAMPLING_RATE_HZ * 5  # 5 seconds buffer
            self.daq_task.timing.cfg_samp_clk_timing(
                rate=self.config.DAQ_SAMPLING_RATE_HZ,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=buffer_size
            )
            
            self.daq_reader = nidaqmx.stream_readers.AnalogMultiChannelReader(self.daq_task.in_stream)
            self.daq_task.start()
            
            print(f"DAQ initialized with {self.num_channels} channels at {self.config.DAQ_SAMPLING_RATE_HZ} Hz")
            return True
            
        except Exception as e:
            print(f"Error initializing DAQ: {e}")
            if self.daq_task:
                try:
                    self.daq_task.close()
                except:
                    pass
                self.daq_task = None
                self.daq_reader = None
            return False
    
    def _cleanup_daq(self):
        """Clean up DAQ resources"""
        if self.daq_task:
            try:
                self.daq_task.stop()
                self.daq_task.close()
                print("DAQ task stopped and closed")
            except Exception as e:
                print(f"Error stopping DAQ task: {e}")
            finally:
                self.daq_task = None
                self.daq_reader = None
    
    def _simulate_emg_data(self):
        """Generate simulated EMG data for testing"""
        t = time.time()
        data = np.zeros((self.num_channels, self.config.CHUNK_SIZE))
        
        # Base frequency and varying amplitude
        base_freq = 2.0  # Hz
        amplitude = 0.5 + 0.5 * np.sin(t / 5)  # Amplitude varies over time
        
        for i in range(self.num_channels):
            # Each channel gets slightly different frequency
            freq = base_freq * (1.0 + 0.2 * i)
            for j in range(self.config.CHUNK_SIZE):
                # Add baseline noise
                noise = 0.1 * np.random.randn()
                # Add sinusoidal component
                signal = amplitude * np.sin(2 * np.pi * freq * (t - j * 0.001)) + noise
                data[i, j] = signal
            
        return data
    
    def _read_daq_data(self):
        """Read a chunk of data from the DAQ"""
        if self.config.USE_SIMULATION:
            return self._simulate_emg_data()
        
        try:
            data_buffer = np.zeros((self.num_channels, self.config.CHUNK_SIZE))
            self.daq_reader.read_many_sample(
                data_buffer,
                number_of_samples_per_channel=self.config.CHUNK_SIZE,
                timeout=1.0
            )
            return data_buffer
        except Exception as e:
            print(f"Error reading DAQ data: {e}")
            return None
    
    def _send_data(self, status, intensity, data, movement="Rest", confidence=0.0):
        """
        Send EMG data and classification through ZMQ socket
        
        Args:
            status: EMGStatus (IDLE or ACTIVE)
            intensity: Signal intensity (0-1)
            data: Raw EMG data
            movement: Detected movement type
            confidence: Classification confidence
        """
        # For plot_data, we'll send a subset of the first channel
        plot_data = data[0, :].tolist()
        
        # Prepare the message with enhanced movement information
        message = {
            'status': status,
            'intensity': float(intensity),
            'plot_data': plot_data,
            'timestamp': time.time(),
            'movement': movement,
            'confidence': float(confidence),
        }
        
        # Send the message
        try:
            self.socket.send_string(json.dumps(message))
            return True
        except Exception as e:
            print(f"Error sending ZMQ message: {e}")
            return False
    
    def run(self):
        """Main processing loop"""
        if not self._init_daq():
            print("Failed to initialize DAQ, exiting")
            return
        
        self.running = True
        print("Starting DAQ-to-ZMQ bridge...")
        
        # Calibration phase - collect 2 seconds of data for baseline
        print("Starting 2-second calibration phase...")
        calibration_data = []
        calibration_start_time = time.time()
        
        # Collect 2 seconds of calibration data (assuming patient at rest)
        while time.time() - calibration_start_time < 2.0 and self.running:
            data = self._read_daq_data()
            if data is not None:
                calibration_data.append(data)
                time.sleep(0.01)
        
        # Process calibration data to estimate DC offsets
        if calibration_data:
            all_calibration = np.concatenate(calibration_data, axis=1)
            self.metadata['dc_offsets'] = np.mean(all_calibration, axis=1)
            print(f"Calibration complete. DC offsets: {self.metadata['dc_offsets']}")
        
        # Main loop
        warm_up_count = 0  # For initial warm-up period before predictions start
        warm_up_needed = 10  # How many chunks needed before starting predictions
        
        try:
            # Main processing loop
            while self.running:
                loop_start = time.time()
                
                # Read data from DAQ
                data = self._read_daq_data()
                if data is None:
                    time.sleep(0.01)
                    continue
                
                # Apply filters
                filtered_data = self._apply_filters(data)
                
                # Update the buffer with new data
                self._update_buffer(filtered_data)
                
                # Initial warm-up period
                if warm_up_count < warm_up_needed:
                    warm_up_count += 1
                    intensity = 0.0  # No movement during warm-up
                    status = EMGStatus.IDLE
                    movement = "Rest"
                    confidence = 1.0
                    if warm_up_count == warm_up_needed:
                        print("Warm-up complete, starting movement detection...")
                else:
                    # Calculate features and get prediction
                    intensity, movement, confidence = self._calculate_features_and_predict(filtered_data)
                    
                    # Determine EMG status
                    status = self._determine_status(intensity, movement, confidence)
                
                # Send the data
                self._send_data(status, intensity, filtered_data, movement, confidence)
                
                # Control loop timing
                elapsed = time.time() - loop_start
                sleep_time = max(0.001, (self.config.CHUNK_DURATION_MS / 1000) - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nStopping DAQ-to-ZMQ bridge...")
        except Exception as e:
            print(f"Error in DAQ-to-ZMQ bridge: {e}")
        finally:
            self.running = False
            self._cleanup_daq()
            self.socket.close()
            self.context.term()
            print("DAQ-to-ZMQ bridge stopped")
        
    def stop(self):
        """Stop the bridge"""
        self.running = False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DAQ-to-ZMQ Bridge for EMG Data")
    parser.add_argument("--simulate", action="store_true", help="Use simulated EMG data")
    parser.add_argument("--device", type=str, default=Config.DAQ_DEVICE_NAME, help="DAQ device name")
    parser.add_argument("--channels", type=str, default=Config.DAQ_CHANNELS_STR, help="DAQ channel string (e.g., ai0:3)")
    parser.add_argument("--rate", type=int, default=Config.DAQ_SAMPLING_RATE_HZ, help="Sampling rate in Hz")
    parser.add_argument("--port", type=int, default=Config.ZMQ_PUBLISHER_PORT, help="ZMQ publisher port")
    parser.add_argument("--no-filter", action="store_true", help="Disable EMG filtering")
    parser.add_argument("--no-classifier", action="store_true", help="Disable advanced EMG classifier")
    parser.add_argument("--model", type=str, default=Config.CLASSIFIER_MODEL_PATH, help="Path to classifier model")
    
    args = parser.parse_args()
    
    # Apply command line arguments to config
    config = Config()
    config.USE_SIMULATION = args.simulate or not NIDAQMX_AVAILABLE
    config.DAQ_DEVICE_NAME = args.device
    config.DAQ_CHANNELS_STR = args.channels
    config.DAQ_SAMPLING_RATE_HZ = args.rate
    config.ZMQ_PUBLISHER_PORT = args.port
    config.APPLY_FILTERS = not args.no_filter
    config.USE_ADVANCED_CLASSIFIER = not args.no_classifier
    config.CLASSIFIER_MODEL_PATH = args.model
    
    # Initialize and run the bridge
    bridge = DAQtoZMQBridge(config)
    try:
        bridge.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()