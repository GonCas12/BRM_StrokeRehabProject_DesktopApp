#!/usr/bin/env python3
"""
Test EMG classifier with a sequence of real EMG recordings
"""

import sys
import os
import numpy as np
import time
import threading
import keyboard  # You may need to install this: pip install keyboard

# Insert the mock_nidaqmx module first in sys.modules
import mock_nidaqmx_module
sys.modules['nidaqmx'] = mock_nidaqmx_module

# Now import the bridge which will use the mock DAQ
from daq_to_zmq_bridge import DAQtoZMQBridge, Config

# Define the sequence of EMG files to test
EMG_SEQUENCE = [
    {
        "file": "SS01_T1_emgData_Rest_Biceps_FDS_20250523_113024.npy",
        "movement": "Rest",
        "key": "1"  # Press 1 to switch to this recording
    },
    {
        "file": "SS01_T7_emgData_Flexion_Biceps_FDS_20250523_113319.npy",
        "movement": "Flexion",
        "key": "2"  # Press 2 to switch to this recording
    },
    {
        "file": "SS01_T1_emgData_Extension_Biceps_FDS_20250523_113640.npy",
        "movement": "Extension",
        "key": "3"  # Press 3 to switch to this recording
    }
]

# Current EMG file index
current_index = 0

def switch_emg_file(index):
    """Switch to a different EMG file"""
    global current_index
    if 0 <= index < len(EMG_SEQUENCE):
        file = EMG_SEQUENCE[index]["file"]
        if os.path.exists(file):
            current_index = index
            mock_nidaqmx_module.PRE_RECORDED_DATA_FILE = file
            print(f"Switched to EMG file: {file} ({EMG_SEQUENCE[index]['movement']})")
            
            # Reset the mock DAQ reader's sample index to start from the beginning
            if hasattr(mock_nidaqmx_module, "reset_reader_state"):
                mock_nidaqmx_module.reset_reader_state()
            return True
        else:
            print(f"Error: EMG file not found: {file}")
    return False

# Function to monitor keypress and switch EMG files
def monitor_keypresses():
    print("\nKeyboard controls:")
    for entry in EMG_SEQUENCE:
        print(f"  Press {entry['key']} for {entry['movement']} ({entry['file']})")
    print("  Press q to quit")
    
    def on_key_event(e):
        if e.event_type == keyboard.KEY_DOWN:
            if e.name == 'q':
                print("Quitting...")
                os._exit(0)
            
            # Check for number keys
            for i, entry in enumerate(EMG_SEQUENCE):
                if e.name == entry["key"]:
                    switch_emg_file(i)
                    break
    
    keyboard.hook(on_key_event)

def on_classification(movement, confidence):
    """Print classification results"""
    print(f"Classified movement: {movement} with confidence: {confidence:.2f}")

# Check that files exist
for entry in EMG_SEQUENCE:
    if not os.path.exists(entry["file"]):
        print(f"Warning: EMG file not found: {entry['file']}")

# Start with the first file
if not switch_emg_file(0):
    print("Error: Failed to set initial EMG file. Please check file paths.")
    sys.exit(1)

# Configure the bridge
config = Config()
config.DAQ_SAMPLING_RATE_HZ = 2000  # Match your recording sampling rate
config.USE_ADVANCED_CLASSIFIER = True
config.CLASSIFIER_MODEL_PATH = "best_temporal_emg_model.pkl"  # Your classifier model
config.DAQ_CHANNELS_STR = "ai0:1"  # 2 channels
config.classification_callback = on_classification  # Add callback to see classifications

# Add a reset function to mock_nidaqmx_module
def reset_reader_state():
    """Reset the MockAnalogMultiChannelReader state to start from beginning"""
    for task in mock_nidaqmx_module.__dict__.values():
        if hasattr(task, 'in_stream') and hasattr(task.in_stream, 'current_sample_index'):
            task.in_stream.current_sample_index = 0
            print("Reset mock DAQ reader sample index")

mock_nidaqmx_module.reset_reader_state = reset_reader_state

# Start keyboard monitoring in a thread
threading.Thread(target=monitor_keypresses, daemon=True).start()

# Initialize and run the bridge
bridge = DAQtoZMQBridge(config)

print("\n=== Starting DAQ-to-ZMQ bridge with real EMG sequence ===")
print("Press keys 1-3 to switch between movements, q to quit\n")

try:
    bridge.run()
except KeyboardInterrupt:
    print("\nTest stopped by user")
finally:
    bridge.stop()