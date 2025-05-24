# mock_nidaqmx_module.py
import numpy as np
import time
import os
import re # For parsing channel strings
from emg_debug_logger import logger

# --- Configuration for Mock DAQ ---
MOCK_SAMPLING_RATE = 2000
MOCK_NUM_CHANNELS_TO_SIMULATE_DEFAULT = 2
MOCK_VOLTAGE_MIN = -5.0
MOCK_VOLTAGE_MAX = 5.0
PRE_RECORDED_DATA_FILE = "emg_sequence_test.npy" # e.g., shape (num_channels, total_samples)
_LAST_LOADED_FILE = None
_PRE_RECORDED_DATA_CACHE = None
# --- End Mock DAQ Configuration ---

class AcquisitionType: CONTINUOUS = 10123
class TerminalConfiguration: DEFAULT = -1; RSE = 10083; NRSE = 10078; DIFF = 10106

class MockNIDAQmxError(Exception):
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code
        self.error_message = message
class errors: DaqError = MockNIDAQmxError

def reload_data_file():
    """Force reload of the pre-recorded data file"""
    global _LAST_LOADED_FILE, _PRE_RECORDED_DATA_CACHE
    
    if PRE_RECORDED_DATA_FILE and os.path.exists(PRE_RECORDED_DATA_FILE):
        try:
            # Only reload if file has changed or not loaded yet
            if _LAST_LOADED_FILE != PRE_RECORDED_DATA_FILE or _PRE_RECORDED_DATA_CACHE is None:
                print(f"Mock DAQ: Loading pre-recorded data from '{PRE_RECORDED_DATA_FILE}'")
                _PRE_RECORDED_DATA_CACHE = np.load(PRE_RECORDED_DATA_FILE)
                _LAST_LOADED_FILE = PRE_RECORDED_DATA_FILE
                print(f"Mock DAQ: Loaded data with shape {_PRE_RECORDED_DATA_CACHE.shape}")
                
                # Reset all readers that might be using this data
                for name, obj in list(globals().items()):
                    if isinstance(obj, MockAnalogMultiChannelReader_Internal):
                        obj.pre_recorded_data = _PRE_RECORDED_DATA_CACHE
                        obj.current_sample_index = 0
                        print(f"Mock DAQ: Reset reader {name}")
                
                return True
        except Exception as e:
            print(f"Mock DAQ ERROR: Could not load pre-recorded data '{PRE_RECORDED_DATA_FILE}': {e}.")
            _PRE_RECORDED_DATA_CACHE = None
            _LAST_LOADED_FILE = None
    return False


class MockAnalogMultiChannelReader_Internal:
    def __init__(self, task_instance):
        self.task = task_instance
        self.pre_recorded_data = None
        self.current_sample_index = 0
        self.last_read_time = time.monotonic()
        self._preload_validated = False

        # --- For more continuous synthetic data ---
        self.synthetic_data_offset = 0 # To make synthetic data scroll
        self.synthetic_buffer_len = MOCK_SAMPLING_RATE * 5 # Generate 5s of synthetic data at a time
        self.current_synthetic_data = None
        # --- End synthetic data state ---


        if PRE_RECORDED_DATA_FILE and os.path.exists(PRE_RECORDED_DATA_FILE):
            try:
                self.pre_recorded_data = np.load(PRE_RECORDED_DATA_FILE)
                logger.info(f"Mock DAQ: Loaded pre-recorded data from '{PRE_RECORDED_DATA_FILE}'")
                print(f"Mock DAQ: Attempted to load pre-recorded data ({self.pre_recorded_data.shape}) from '{PRE_RECORDED_DATA_FILE}'")
            except Exception as e:
                print(f"Mock DAQ WARNING: Could not load pre-recorded data '{PRE_RECORDED_DATA_FILE}': {e}.")
                self.pre_recorded_data = None
        # Use the global cache if available
        global _PRE_RECORDED_DATA_CACHE
        if _PRE_RECORDED_DATA_CACHE is not None:
            self.pre_recorded_data = _PRE_RECORDED_DATA_CACHE
            print(f"Mock DAQ: Using cached pre-recorded data ({self.pre_recorded_data.shape})")
            logger.info(f"Mock DAQ: Using cached pre-recorded data ({self.pre_recorded_data.shape})")
        else:
            # Load from file
            reload_data_file()
            self.pre_recorded_data = _PRE_RECORDED_DATA_CACHE
            logger.info(f"Mock DAQ: Using pre-recorded data from file '{PRE_RECORDED_DATA_FILE}'")

        if self.pre_recorded_data is None:
            print(f"Mock DAQ: Pre-recorded data will not be used (either not found, load failed, or not specified). Will generate synthetic data.")

    def reset_state(self):
        """Reset the reader state to start from the beginning of the data file"""
        self.current_sample_index = 0
        self.last_read_time = time.monotonic()
        self._preload_validated = False
        print(f"MockReader reset to start of {PRE_RECORDED_DATA_FILE}")

    def _validate_preload_channels(self):
        if self.pre_recorded_data is not None:
            if self.pre_recorded_data.shape[0] != self.task.number_of_channels:
                print(f"Mock DAQ WARNING: Pre-recorded data has {self.pre_recorded_data.shape[0]} channels, "
                      f"but task is configured for {self.task.number_of_channels}. Discarding pre-recorded data.")
                self.pre_recorded_data = None
            else:
                print(f"Mock DAQ: Pre-recorded data validated for {self.task.number_of_channels} channels.")
        self._preload_validated = True

    def _generate_new_synthetic_chunk(self, num_channels, num_samples_to_generate):
        """Generates a new chunk of somewhat continuous synthetic data."""
        # Base noise
        chunk = np.random.normal(0, 0.05, (num_channels, num_samples_to_generate))
        # Add some activity to one channel at a time, cycling through channels
        # Or make it more random
        if np.random.rand() < 0.3: # 30% chance of a burst somewhere in this larger chunk
            burst_channel = np.random.randint(0, num_channels)
            total_len = num_samples_to_generate
            
            # Create a longer burst within the synthetic buffer
            burst_start_rel = np.random.randint(0, total_len // 3)
            burst_len = np.random.randint(total_len // 4, total_len // 2 + 1)
            burst_end_rel = min(burst_start_rel + burst_len, total_len)
            
            amplitude = np.random.uniform(0.4, 1.5)
            frequency = np.random.uniform(10, 60) # Lower frequencies for longer windows
            t_burst = np.linspace(0, (burst_end_rel - burst_start_rel) / self.task.sampling_rate,
                                  burst_end_rel - burst_start_rel, endpoint=False)
            burst_signal = amplitude * np.sin(2 * np.pi * frequency * t_burst)
            burst_signal += np.random.normal(0, 0.1, len(burst_signal)) # Noise on burst

            chunk[burst_channel, burst_start_rel:burst_end_rel] += burst_signal
        return np.clip(chunk, self.task.min_val, self.task.max_val)


    def read_many_sample(self, data_buffer, number_of_samples_per_channel, timeout=1.0):
        num_ch = self.task.number_of_channels

        expected_time = number_of_samples_per_channel / self.task.sampling_rate
        current_time = time.monotonic()
        time_lapsed = current_time - self.last_read_time
        if time_lapsed < expected_time: time.sleep(max(0, expected_time - time_lapsed))
        self.last_read_time = time.monotonic()

        if not self._preload_validated: self._validate_preload_channels()

        if self.pre_recorded_data is not None:
            # ... (pre-recorded data logic - keep as is, with looping) ...
            remaining_samples_in_file = self.pre_recorded_data.shape[1] - self.current_sample_index
            samples_to_provide = min(number_of_samples_per_channel, remaining_samples_in_file)

            if samples_to_provide > 0:
                chunk = self.pre_recorded_data[:, self.current_sample_index : self.current_sample_index + samples_to_provide]
                data_buffer[:, :samples_to_provide] = chunk
                self.current_sample_index += samples_to_provide
                if self.current_sample_index >= self.pre_recorded_data.shape[1]:
                    print("Mock DAQ: Reached end of pre-recorded data. Looping.")
                    self.current_sample_index = 0
                if samples_to_provide < number_of_samples_per_channel:
                    padding = number_of_samples_per_channel - samples_to_provide
                    noise_pad = np.random.normal(0, 0.01, (num_ch, padding))
                    data_buffer[:, samples_to_provide:] = noise_pad
                return number_of_samples_per_channel
            else: # Should only happen if pre_recorded_data is empty or too short
                self.pre_recorded_data = None # Fallback to synthetic
                print("Mock DAQ: Pre-recorded data insufficient or exhausted, switching to synthetic.")


        # --- Synthetic Data Generation (Improved for more continuity) ---
        if self.current_synthetic_data is None or self.synthetic_data_offset >= self.current_synthetic_data.shape[1]:
            # Generate a new large block of synthetic data
            self.current_synthetic_data = self._generate_new_synthetic_chunk(num_ch, self.synthetic_buffer_len)
            self.synthetic_data_offset = 0
            print(f"Mock DAQ: Generated new synthetic data block ({self.current_synthetic_data.shape}).")

        # Provide a slice from the current synthetic block
        samples_available_in_block = self.current_synthetic_data.shape[1] - self.synthetic_data_offset
        samples_to_serve = min(number_of_samples_per_channel, samples_available_in_block)

        chunk_to_serve = self.current_synthetic_data[:, self.synthetic_data_offset : self.synthetic_data_offset + samples_to_serve]
        data_buffer[:, :samples_to_serve] = chunk_to_serve
        self.synthetic_data_offset += samples_to_serve

        # If requested more than available in current block (should be rare with large synthetic_buffer_len)
        if samples_to_serve < number_of_samples_per_channel:
            remaining_needed = number_of_samples_per_channel - samples_to_serve
            # This case implies we need to regenerate immediately, or that requested samples > synthetic_buffer_len
            # For simplicity, just pad with new noise for the remainder of this call
            print(f"Mock DAQ: Synthetic block short, padding {remaining_needed} samples.")
            padding_data = self._generate_new_synthetic_chunk(num_ch, remaining_needed)
            data_buffer[:, samples_to_serve:] = padding_data
        return number_of_samples_per_channel

class stream_readers:
    @staticmethod
    def AnalogMultiChannelReader(task_in_stream): # task_in_stream is actually task.in_stream object
        # The task_in_stream passed from the main script is actually the
        # task.in_stream object, which is an instance of MockAnalogMultiChannelReader_Internal
        # if task.in_stream was set up correctly.
        # However, the main script calls nidaqmx.stream_readers.AnalogMultiChannelReader(task.in_stream)
        # So this function should return an instance of our internal reader.
        # The task_in_stream argument here is actually the `task.in_stream` object,
        # which is already an instance of MockAnalogMultiChannelReader_Internal.
        # This is a bit circular but makes the external call work.
        # A cleaner way might be to have task.in_stream be a simple placeholder,
        # and this factory method creates the actual reader.

        # Let's assume task_in_stream is the task._in_stream_proxy attribute
        # which holds the task instance itself.
        if isinstance(task_in_stream, Task._InStreamProxy):
            return MockAnalogMultiChannelReader_Internal(task_in_stream.task)
        else:
            # Fallback or error if the structure isn't as expected
            # This path shouldn't usually be taken if MockTask is set up correctly
            print("Mock DAQ WARNING: AnalogMultiChannelReader received unexpected task_in_stream type.")
            # Attempt to find the task instance if task_in_stream is the reader itself
            if hasattr(task_in_stream, 'task') and isinstance(task_in_stream.task, Task):
                 return task_in_stream # It's already a reader instance
            raise MockNIDAQmxError("Cannot correctly initialize MockAnalogMultiChannelReader.")
# **** END MOCK stream_readers module ****

class MockAIChannels:
    # ... (Keep MockAIChannels as is from your provided code) ...
    def __init__(self, task_instance):
        self.task = task_instance
    def add_ai_voltage_chan(self, physical_channel_name, terminal_config=None, min_val=-5.0, max_val=5.0, **kwargs):
        try:
            dev_chan_part = physical_channel_name.split('/')[-1]
            if ':' in dev_chan_part:
                chan_parts = dev_chan_part.split(':'); start_chan_str = re.findall(r'ai(\d+)', chan_parts[0])
                if not start_chan_str: raise ValueError("Could not parse start channel")
                start_chan_num = int(start_chan_str[0]); end_chan_num = int(chan_parts[1])
                self.task.number_of_channels = end_chan_num - start_chan_num + 1
            elif ',' in dev_chan_part: self.task.number_of_channels = len(dev_chan_part.split(','))
            else: self.task.number_of_channels = 1
        except Exception as e:
            print(f"Mock DAQ WARNING: Parse error for '{physical_channel_name}': {e}. Defaulting to {MOCK_NUM_CHANNELS_TO_SIMULATE_DEFAULT}.")
            self.task.number_of_channels = MOCK_NUM_CHANNELS_TO_SIMULATE_DEFAULT
        self.task.min_val = min_val; self.task.max_val = max_val
        print(f"Mock DAQ: Task for {self.task.number_of_channels} channels. Range: {min_val} to {max_val} V.")
        if hasattr(self.task.in_stream, '_validate_preload_channels'): self.task.in_stream._preload_validated = False

class MockTiming:
    # ... (Keep MockTiming as is) ...
    def __init__(self, task_instance): self.task = task_instance
    def cfg_samp_clk_timing(self, rate, sample_mode=None, samps_per_chan=None, **kwargs):
        self.task.sampling_rate = rate
        print(f"Mock DAQ: Sampling rate set to {self.task.sampling_rate} Hz.")

class Task:
    class _InStreamProxy: # Helper class to pass task instance to reader factory
        def __init__(self, task):
            self.task = task
    # ... (Keep Task as is, ensure in_stream gets an instance of the updated MockAnalogMultiChannelReader) ...
    def __init__(self, new_task_name=""):
        print("Mock DAQ: Task object initialized.")
        self.number_of_channels = MOCK_NUM_CHANNELS_TO_SIMULATE_DEFAULT
        self.sampling_rate = MOCK_SAMPLING_RATE
        self.min_val = MOCK_VOLTAGE_MIN; self.max_val = MOCK_VOLTAGE_MAX
        self._is_running = False
        self.ai_channels = MockAIChannels(self); self.timing = MockTiming(self)
        self.in_stream = MockAnalogMultiChannelReader_Internal(self) # Ensures updated reader is used

    def start(self):
        self._is_running = True
        if hasattr(self.in_stream, 'current_sample_index'): self.in_stream.current_sample_index = 0
        if hasattr(self.in_stream, 'last_read_time'): self.in_stream.last_read_time = time.monotonic()
        # Reset synthetic data generation state
        if hasattr(self.in_stream, 'current_synthetic_data'): self.in_stream.current_synthetic_data = None
        if hasattr(self.in_stream, 'synthetic_data_offset'): self.in_stream.synthetic_data_offset = 0
        print("Mock DAQ: Task started.")

    def stop(self): self._is_running = False; print("Mock DAQ: Task stopped.")
    def close(self): self._is_running = False; print("Mock DAQ: Task closed.")
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()