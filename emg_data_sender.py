import zmq
import time
import json
import numpy as np
import random

class EMGDataSender:
    def __init__(self, port=5555):
        """Initialize a ZMQ server to send EMG data"""
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        print(f"EMG data sender started on port {self.port}")
        # Give the socket some time to bind
        time.sleep(0.5)
    
    def send_data(self, status, intensity, plot_data):
        """Send EMG data through the ZMQ socket"""
        data = {
            'status': status,
            'intensity': intensity,
            'plot_data': plot_data,
            'timestamp': time.time()
        }
        message = json.dumps(data)
        self.socket.send_string(message)
    
    def close(self):
        """Close the ZMQ socket"""
        self.socket.close()
        self.context.term()
        print("EMG data sender stopped")

# Example usage
if __name__ == "__main__":
    sender = EMGDataSender()
    
    # Define status options to match what the app expects
    POSSIBLE_STATUSES = ['IDLE', 'CORRECT_WEAK', 'CORRECT_STRONG', 'INCORRECT']
    STATUS_WEIGHTS = [0.4, 0.3, 0.2, 0.1]
    
    try:
        print("Sending EMG data. Press Ctrl+C to stop.")
        while True:
            # Choose a status based on weights
            status = random.choices(POSSIBLE_STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
            
            # Generate intensity based on status
            intensity = 0.0
            if status == 'CORRECT_WEAK':
                intensity = random.uniform(0.2, 0.5)
            elif status == 'CORRECT_STRONG':
                intensity = random.uniform(0.55, 1.0)
            
            # Generate simulated EMG data
            plot_data = []
            for i in range(100):
                base_signal = random.gauss(0, 0.1)
                if status.startswith('CORRECT'):
                    base_signal += intensity * 0.5 * np.sin(i * 0.2)
                plot_data.append(base_signal)
            
            # Send the data
            sender.send_data(status, intensity, plot_data)
            print(f"Sent data with status: {status}, intensity: {intensity:.2f}")
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping sender...")
    finally:
        sender.close()