import numpy as np
import time
import threading
import joblib
from scipy import signal
from PySide6.QtCore import QObject, Signal, QThread, QTimer, Slot
import mock_nidaqmx_module as nidaqmx

def preprocess_emg(emg_data, metadata, apply_filters=True):
    """
    Preprocess EMG data using metadata for offset correction - UPDATED VERSION
    """
    n_channels, n_samples = emg_data.shape
    processed_data = np.zeros_like(emg_data)
    
    # 1. Apply DC offset correction using metadata
    dc_offsets = metadata['dc_offsets']
    sampling_rate = metadata['sampling_rate']

    # Handle sampling rate conversion (from numpy array to scalar if needed)
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
        channel_data = channel_data - dc_offset
        
        if apply_filters:
            # Apply bandpass filter
            channel_data = signal.sosfilt(sos_bandpass, channel_data)
            
            # Apply notch filter
            channel_data = signal.sosfilt(sos_notch, channel_data)
        
        # Store processed data
        processed_data[c, :] = channel_data
    
    return processed_data

def apply_bandpass_filter(signal_data, low_freq, high_freq, sampling_rate):
    """Apply bandpass filter to signal - NEW FUNCTION"""
    try:
        nyquist = 0.5 * sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure filter parameters are valid
        if low >= 1.0 or high >= 1.0:
            low = min(0.95, low)
            high = min(0.99, high)
        
        if low >= high:
            return signal_data  # Return original if invalid range
            
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        filtered = signal.sosfilt(sos, signal_data)
        return filtered
    except Exception as e:
        print(f"Bandpass filter failed: {e}")
        return signal_data  # Return original if filter fails

def enhanced_rms_features(signal_data, sampling_rate):
    """Extract RMS in different frequency bands - NEW FUNCTION"""
    try:
        # Low band (20-60 Hz) - slow motor unit activity
        low_filtered = apply_bandpass_filter(signal_data, 20, 60, sampling_rate)
        rms_low = np.sqrt(np.mean(low_filtered**2))
        
        # Mid band (60-150 Hz) - main EMG content
        mid_filtered = apply_bandpass_filter(signal_data, 60, 150, sampling_rate)
        rms_mid = np.sqrt(np.mean(mid_filtered**2))
        
        # High band (150-450 Hz) - fast motor unit activity
        high_filtered = apply_bandpass_filter(signal_data, 150, 450, sampling_rate)
        rms_high = np.sqrt(np.mean(high_filtered**2))
        
        return [rms_low, rms_mid, rms_high]
    except Exception as e:
        print(f"Enhanced RMS features failed: {e}")
        return [0.0, 0.0, 0.0]

def cross_channel_features(window):
    """Calculate cross-channel correlation features - NEW FUNCTION"""
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
    Extract enhanced features from segmented EMG data with temporal context - UPDATED VERSION
    """
    if windows.shape[0] == 0:
        return np.array([])

    n_windows, n_data_channels, window_size = windows.shape
    epsilon = 1e-10

    # Define enhanced features per channel - UPDATED COUNTING
    num_basic_features_per_channel = 12  # Original features
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
            
            mean_freq, median_freq, peak_freq = 0, 0, 0
            low_band_power, med_band_power, high_band_power = 0, 0, 0

            if window_size > 1:
                try:
                    freqs = np.fft.rfftfreq(window_size, d=1.0/sampling_rate)
                    if len(signal_data) == window_size:
                        ps = np.abs(np.fft.rfft(signal_data))**2
                        total_power = np.sum(ps)
                    
                        if total_power > 0:
                            mean_freq = np.sum(freqs * ps) / total_power
                            
                            cum_sum_ps = np.cumsum(ps)
                            median_idx_arr = np.where(cum_sum_ps >= total_power/2)[0]
                            if len(median_idx_arr) > 0:
                                median_freq = freqs[median_idx_arr[0]]
                            
                            if len(ps) > 0:
                                peak_freq_idx = np.argmax(ps)
                                if ps[peak_freq_idx] > 0:
                                    peak_freq = freqs[peak_freq_idx]
                            
                            low_band_mask = (freqs >= 20) & (freqs <= 100)
                            med_band_mask = (freqs > 100) & (freqs <= 250)
                            high_band_mask = (freqs > 250) & (freqs <= 450)
                            
                            low_band_power = np.sum(ps[low_band_mask]) / total_power
                            med_band_power = np.sum(ps[med_band_mask]) / total_power
                            high_band_power = np.sum(ps[high_band_mask]) / total_power
                except:
                    pass  # Keep zeros if FFT fails
            
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

def generate_complete_feature_names(n_channels, basic_features, include_context=True):
    """
    Generate complete feature names including cross-channel correlations - NEW FUNCTION
    """
    feature_names = []
    
    # 1. Basic features per channel
    for c in range(n_channels):
        channel_prefix = f"Ch{c+1}_"
        for f_name in basic_features:
            feature_names.append(f"{channel_prefix}{f_name}")
    
    # 2. Cross-channel correlation features
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

def get_feature_names_for_data(n_channels):
    """
    Get properly ordered feature names that match extract_features_with_context output - NEW FUNCTION
    """
    basic_features = [
        "MAV", "RMS", "ZC", "WL", "Entropy", "Variance",
        "MeanFreq", "MedianFreq", "PeakFreq", 
        "LowBandPower", "MedBandPower", "HighBandPower",
        "RMS_Low", "RMS_Mid", "RMS_High",  # Enhanced RMS features
    ]
    
    return generate_complete_feature_names(n_channels, basic_features, include_context=True)

class EMGClassifier:
    def __init__(self, model_path="best_temporal_emg_model.pkl"):
        print(f"Loading model from {model_path}")
        
        # Check if model file exists
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            model_data = joblib.load(model_path)
            
            # Validate model structure
            if not isinstance(model_data, dict):
                raise ValueError("Model file does not contain expected dictionary structure")
            
            # Extract model components with validation
            if 'pipeline' not in model_data:
                raise ValueError("Model file missing 'pipeline' key")
            
            self.pipeline = model_data['pipeline']
            self.window_size = model_data.get('window_size', 5)
            self.min_confidence = model_data.get('min_confidence', 0.5)
            self.label_names = model_data.get('label_names', ['Rest', 'Flexion', 'Extension'])
            self.history_size = model_data.get('history_size', 3)
            self.feature_names = model_data.get('feature_names', [])
            
            print(f"Model loaded successfully with labels: {self.label_names}")
            print(f"Window size: {self.window_size}, Min confidence: {self.min_confidence}")
            print(f"History size: {self.history_size}")
            
            # Determine expected feature count
            # Try to infer from model or feature names
            if self.feature_names:
                expected_features = len(self.feature_names)
                print(f"Expected features from model: {expected_features}")
            else:
                print("Warning: No feature names found in model")
                expected_features = None
            
            # Test the model with dummy data to ensure it works
            if expected_features:
                dummy_features = np.zeros((1, expected_features))
            else:
                # Fallback: try different feature counts
                for test_count in [60, 120, 240]:  # Common feature counts
                    try:
                        dummy_features = np.zeros((1, test_count))
                        test_prediction = self.pipeline.predict(dummy_features)
                        expected_features = test_count
                        print(f"Model accepts {test_count} features")
                        break
                    except:
                        continue
                
                if expected_features is None:
                    raise ValueError("Could not determine expected feature count for model")
            
            try:
                test_prediction = self.pipeline.predict(dummy_features)
                print(f"Model test successful. Test prediction: {test_prediction}")
            except Exception as e:
                print(f"Warning: Model test failed: {e}")
                raise
            
            self.expected_features = expected_features
            
            # Initialize other attributes
            self.buffer_size = 500
            self.buffer = None
            self.buffer_filled = False
            self.samples_collected = 0
            
            self.prediction_history = []
            self.probability_history = []
            self.current_state = "Rest"
            self.current_confidence = 0.0
            self.feature_history = []
            
        except Exception as e:
            print(f"Detailed error loading model: {e}")
            print(f"Model file size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'} bytes")
            raise
    
    def initialize_buffer(self, n_channels):
        self.buffer = np.zeros((n_channels, self.buffer_size))
        self.samples_collected = 0
        self.n_channels = n_channels
        print(f"Initialized buffer for {n_channels} channels")
    
    def process_new_data(self, new_data_chunk, metadata):
        try:
            if self.buffer is None:
                self.initialize_buffer(new_data_chunk.shape[0])
            
            chunk_size = new_data_chunk.shape[1]
            
            if not self.buffer_filled:
                space_left = self.buffer_size - self.samples_collected
                samples_to_add = min(chunk_size, space_left)
                
                self.buffer[:, self.samples_collected:self.samples_collected+samples_to_add] = new_data_chunk[:, :samples_to_add]
                self.samples_collected += samples_to_add
                
                if self.samples_collected >= self.buffer_size:
                    self.buffer_filled = True
                    print("Buffer filled, starting classification")
                else:
                    return None, None, None
            else:
                self.buffer = np.roll(self.buffer, -chunk_size, axis=1)
                self.buffer[:, -chunk_size:] = new_data_chunk
            
            # Preprocess the buffer
            processed_data = preprocess_emg(self.buffer, metadata, apply_filters=True)
            signal_intensity = np.sqrt(np.mean(processed_data**2))
            
            # Extract features using the updated function
            window = np.expand_dims(processed_data, axis=0)
            sampling_rate = metadata.get('sampling_rate', 2000)
            
            # Get current window features (without context initially)
            basic_features = extract_features_with_context(window, history_size=0, sampling_rate=sampling_rate)
            
            if basic_features.shape[0] == 0:
                print("Warning: No features extracted")
                return None, None, None
            
            # Store in feature history for temporal context
            self.feature_history.append(basic_features[0])
            if len(self.feature_history) > self.history_size + 1:
                self.feature_history.pop(0)
            
            # Create features with temporal context
            if len(self.feature_history) > self.history_size:
                current_features = self.feature_history[-1]
                history_avg = np.mean(self.feature_history[:-1], axis=0)
                context_features = np.concatenate([current_features, current_features - history_avg])
                features = np.expand_dims(context_features, axis=0)
            else:
                # Pad with zeros for delta features when not enough history
                current_features = self.feature_history[-1]
                features_with_zeros = np.zeros((1, len(current_features) * 2))
                features_with_zeros[0, :len(current_features)] = current_features
                features = features_with_zeros
            
            # Validate feature count
            if features.shape[1] != self.expected_features:
                print(f"Warning: Feature count mismatch. Expected {self.expected_features}, got {features.shape[1]}")
                # Try to handle mismatch
                if features.shape[1] < self.expected_features:
                    # Pad with zeros
                    padded_features = np.zeros((1, self.expected_features))
                    padded_features[0, :features.shape[1]] = features[0]
                    features = padded_features
                else:
                    # Truncate
                    features = features[:, :self.expected_features]
                print(f"Adjusted feature count to {features.shape[1]}")
            
            # Make prediction
            prediction = self.pipeline.predict(features)[0]
            
            # Get confidence
            confidence = None
            if hasattr(self.pipeline.named_steps['classifier'], "predict_proba"):
                try:
                    probabilities = self.pipeline.predict_proba(features)[0]
                    confidence = np.max(probabilities)
                except Exception as e:
                    print(f"Error getting probabilities: {e}")
                    confidence = 0.7
            else:
                confidence = 0.7
            
            # Apply temporal smoothing
            smoothed_prediction, smoothed_confidence = self._apply_smoothing(prediction, confidence)
            
            return smoothed_prediction, smoothed_confidence, signal_intensity
            
        except Exception as e:
            print(f"Error in process_new_data: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def _apply_smoothing(self, prediction, confidence):
        try:
            if confidence is None:
                confidence = 0.5
            
            self.prediction_history.append(prediction)
            self.probability_history.append(confidence)
            
            if len(self.prediction_history) > self.window_size:
                self.prediction_history.pop(0)
                self.probability_history.pop(0)
            
            movement_counts = {}
            for pred in self.prediction_history:
                movement_counts[pred] = movement_counts.get(pred, 0) + 1
            
            if movement_counts:
                most_common = max(movement_counts.items(), key=lambda x: x[1])[0]
                most_common_count = movement_counts[most_common]
                
                avg_confidence = 0.0
                count = 0
                for i, pred in enumerate(self.prediction_history):
                    if pred == most_common:
                        avg_confidence += self.probability_history[i]
                        count += 1
                avg_confidence /= max(1, count)
                
                if most_common != self.current_state:
                    threshold_ratio = 0.3
                    if most_common_count >= len(self.prediction_history) * threshold_ratio and avg_confidence >= self.min_confidence * 0.8:
                        self.current_state = most_common
                        self.current_confidence = avg_confidence
                else:
                    self.current_confidence = avg_confidence
            
            return self.current_state, self.current_confidence
            
        except Exception as e:
            print(f"Error in smoothing: {e}")
            return None, None

class EMGDataAcquisitionWorker(QObject):
    """Worker that runs DAQ in a separate thread"""
    data_ready = Signal(np.ndarray, dict)  # (data_chunk, metadata)
    error_occurred = Signal(str)
    status_changed = Signal(str)
    finished = Signal()
    
    def __init__(self, use_mock=True, mock_file=None):
        super().__init__()
        self.use_mock = use_mock
        self.mock_file = mock_file
        self.running = False
        self.task = None
        self.reader = None
        self.metadata = None
        
    @Slot()
    def start_acquisition(self):
        """Start data acquisition - runs in worker thread"""
        try:
            self.running = True
            self.status_changed.emit("Starting DAQ...")
            
            # Set up DAQ
            if self.use_mock:
                if self.mock_file:
                    nidaqmx.PRE_RECORDED_DATA_FILE = self.mock_file
                    nidaqmx.reload_data_file()
                    print(f"Using mock DAQ with pre-recorded data: {self.mock_file}")
                else:
                    print("Using mock DAQ with simulated data")
            
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan("Dev1/ai0:1", min_val=-5.0, max_val=5.0)
            self.task.timing.cfg_samp_clk_timing(rate=2000, sample_mode=nidaqmx.AcquisitionType.CONTINUOUS)
            self.reader = nidaqmx.stream_readers.AnalogMultiChannelReader(self.task.in_stream)
            
            # Calibration
            self.status_changed.emit("Calibrating...")
            calibration_buffer = np.zeros((2, 2 * 2000))
            self.task.start()
            
            samples_collected = 0
            while samples_collected < calibration_buffer.shape[1] and self.running:
                samples_to_read = min(200, calibration_buffer.shape[1] - samples_collected)
                temp_buffer = np.zeros((2, samples_to_read))
                self.reader.read_many_sample(temp_buffer, samples_to_read)
                calibration_buffer[:, samples_collected:samples_collected+samples_to_read] = temp_buffer
                samples_collected += samples_to_read
            
            dc_offsets = np.mean(calibration_buffer, axis=1)
            self.metadata = {
                'dc_offsets': dc_offsets,
                'sampling_rate': 2000
            }
            
            self.status_changed.emit("Running")
            print(f"DAQ calibration complete. DC offsets: {dc_offsets}")
            
            # Main acquisition loop
            while self.running:
                chunk_size = 200
                temp_buffer = np.zeros((2, chunk_size))
                self.reader.read_many_sample(temp_buffer, chunk_size)
                
                # Emit data to processing thread
                self.data_ready.emit(temp_buffer, self.metadata)
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
                
        except Exception as e:
            error_msg = f"DAQ error: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
        finally:
            self.cleanup()
            self.finished.emit()
    
    @Slot()
    def stop_acquisition(self):
        """Stop data acquisition"""
        print("Stopping DAQ acquisition...")
        self.running = False
    
    def cleanup(self):
        """Clean up DAQ resources"""
        try:
            if self.task:
                self.task.stop()
                self.task.close()
                self.task = None
            self.status_changed.emit("Stopped")
            print("DAQ cleanup completed")
        except Exception as e:
            print(f"Error during DAQ cleanup: {e}")

class EMGProcessingWorker(QObject):
    """Worker that processes EMG data and classifies movements"""
    new_result = Signal(dict)
    movement_detected = Signal(str, float)
    error_occurred = Signal(str)
    finished = Signal()
    
    def __init__(self, model_path="best_temporal_emg_model.pkl"):
        super().__init__()
        self.classifier = None
        self.model_path = model_path
        self.running = True
        
        # Movement tracking
        self.current_movement = "Rest"
        self.movement_start_time = None
        self.movement_threshold = 0.65
        self.min_duration_s = 0.3
        
    @Slot()
    def initialize_classifier(self):
        """Initialize the classifier - called once when starting"""
        try:
            self.classifier = EMGClassifier(self.model_path)
            print("EMG classifier initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize EMG classifier: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
    
    @Slot(np.ndarray, dict)
    def process_data_chunk(self, data_chunk, metadata):
        """Process a chunk of EMG data"""
        if not self.running:
            return
            
        try:
            if not self.classifier:
                return
            
            # Classify the data
            movement, confidence, intensity = self.classifier.process_new_data(data_chunk, metadata)
            
            if movement is not None:
                # Track movement state changes
                timestamp = time.time()
                if movement != self.current_movement:
                    if movement != "Rest" and confidence >= self.movement_threshold:
                        if self.current_movement == "Rest":
                            self.movement_start_time = timestamp
                            print(f"Starting movement: {movement}")
                        else:
                            print(f"Changing movement: {self.current_movement} -> {movement}")
                            if self.movement_start_time is not None:
                                duration = timestamp - self.movement_start_time
                                if duration >= self.min_duration_s:
                                    print(f"Completed movement: {self.current_movement} ({duration:.2f}s)")
                                    self.movement_detected.emit(self.current_movement, confidence)
                            self.movement_start_time = timestamp
                            
                    elif movement == "Rest" and self.current_movement != "Rest":
                        if self.movement_start_time is not None:
                            duration = timestamp - self.movement_start_time
                            if duration >= self.min_duration_s:
                                print(f"Completed movement: {self.current_movement} ({duration:.2f}s)")
                                self.movement_detected.emit(self.current_movement, confidence)
                        self.movement_start_time = None
                    
                    self.current_movement = movement
                
                # Determine status for the app
                if movement == 'Rest':
                    status = 'NO_MOVEMENT'
                else:
                    if intensity >= 0.5:
                        status = 'CORRECT_STRONG'
                    else:
                        status = 'CORRECT_WEAK'
                
                # Apply DC offset correction for plot data
                plot_data = data_chunk.copy()
                for i in range(data_chunk.shape[0]):
                    plot_data[i, :] = data_chunk[i, :] - metadata['dc_offsets'][i]
                plot_data = plot_data * 1000  # Convert to μV
                
                # Create result
                result = {
                    'status': status,
                    'intensity': float(intensity),
                    'plot_data': plot_data.tolist(),
                    'timestamp': timestamp,
                    'movement': movement,
                    'confidence': float(confidence)
                }
                
                self.new_result.emit(result)
                
        except Exception as e:
            error_msg = f"Error processing EMG data: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
    
    @Slot()
    def stop_processing(self):
        """Stop processing"""
        print("Stopping EMG processing...")
        self.running = False
        self.finished.emit()