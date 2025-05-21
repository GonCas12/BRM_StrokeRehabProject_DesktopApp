import zmq
import time
import json
import threading

class EMGZmqReceiver:
    def __init__(self, port=5555):
        """
        Initialize a ZMQ server to receive EMG data
        
        Args:
            port (int): Port for the ZMQ socket (default: 5555)
        """
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.running = False
        self.latest_data = None
        self.receive_thread = None
    
    def start(self):
        """Start receiving data from the ZMQ socket"""
        self.running = True
        self.receive_thread = threading.Thread(target=self._receive_data)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        print(f"EMG ZMQ receiver started on port {self.port}")
    
    def stop(self):
        """Stop receiving data"""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        self.socket.close()
        self.context.term()
        print("EMG ZMQ receiver stopped")
    
    def _receive_data(self):
        """Receive data from the ZMQ socket in a loop"""
        while self.running:
            try:
                # Set a timeout to allow checking if running changed
                if self.socket.poll(timeout=100) != 0:
                    message = self.socket.recv_string()
                    data = json.loads(message)
                    self.latest_data = data
            except Exception as e:
                print(f"Error receiving EMG data: {e}")
    
    def get_latest_data(self):
        """
        Get the latest received EMG data
        
        Returns:
            dict: Latest EMG data or None if no data received
        """
        return self.latest_data