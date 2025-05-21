#!/usr/bin/env python3
"""
DAQ-to-ZMQ Bridge - Reads EMG data from DAQ and sends it to BRM_StrokeRehab application
"""

import time
import numpy as np
import scipy.signal
import threading
import zmq
import json
import argparse
from enum import Enum

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
    
    def _calculate_features(self, data):
        """Calculate features from EMG data to determine intensity"""
        # Calculate RMS for each channel
        rms_values = np.zeros(self.num_channels)
        
        for i in range(self.num_channels):
            # Root Mean Square
            rms_values[i] = np.sqrt(np.mean(np.square(data[i, :])))
        
        # Get overall intensity (mean of all channels)
        overall_intensity = np.mean(rms_values)
        
        # Normalize to 0-1 range (assuming typical EMG voltage ranges)
        # This may need adjustment based on your specific hardware and subjects
        normalized_intensity = np.clip(overall_intensity / 1.0, 0.0, 1.0)
        
        return normalized_intensity
    
    def _determine_status(self, intensity):
        """Determine EMG status based on intensity"""
        if intensity > self.config.INTENSITY_THRESHOLD:
            return EMGStatus.ACTIVE
        else:
            return EMGStatus.IDLE
    
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
    
    def _send_data(self, status, intensity, data):
        """Send EMG data through ZMQ socket"""
        # For plot_data, we'll send a subset of the first channel
        # The BRM_StrokeRehab app expects plot_data to be a list, not a numpy array
        plot_data = data[0, :].tolist()
        
        message = {
            'status': status,
            'intensity': float(intensity),
            'plot_data': plot_data,
            'timestamp': time.time()
        }
        
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
                
                # Calculate intensity
                intensity = self._calculate_features(filtered_data)
                
                # Determine status
                status = self._determine_status(intensity)
                
                # Send the data
                self._send_data(status, intensity, filtered_data)
                
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
    
    args = parser.parse_args()
    
    # Apply command line arguments to config
    config = Config()
    config.USE_SIMULATION = args.simulate or not NIDAQMX_AVAILABLE
    config.DAQ_DEVICE_NAME = args.device
    config.DAQ_CHANNELS_STR = args.channels
    config.DAQ_SAMPLING_RATE_HZ = args.rate
    config.ZMQ_PUBLISHER_PORT = args.port
    config.APPLY_FILTERS = not args.no_filter
    
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