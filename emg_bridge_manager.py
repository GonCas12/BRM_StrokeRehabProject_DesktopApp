import os
import sys
import time
import subprocess
import signal
import threading
from PyQt5.QtCore import QObject, pyqtSignal as Signal, pyqtSlot as Slot

class EMGBridgeManager(QObject):
    """
    Manages the DAQ-to-ZMQ bridge process from within the application.
    This allows starting and stopping EMG data collection from the GUI.
    """
    status_changed = Signal(str)  # Signal to update UI with bridge status
    
    def __init__(self, bridge_script_path=None):
        super().__init__()
        # Find the bridge script path
        if bridge_script_path is None:
            # Assume the bridge script is in the same directory as this file
            self.bridge_script_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "daq_to_zmq_bridge.py"
            )
        else:
            self.bridge_script_path = bridge_script_path
            
        self.bridge_process = None
        self.is_running = False
        self.monitor_thread = None
        
    def start_bridge(self, simulate=False, device="Dev1", channels="ai0:3", 
                    rate=1000, port=5555, no_filter=False):
        """
        Start the DAQ-to-ZMQ bridge process.
        
        Args:
            simulate (bool): Whether to use simulated EMG data
            device (str): DAQ device name
            channels (str): DAQ channel string (e.g., ai0:3)
            rate (int): Sampling rate in Hz
            port (int): ZMQ publisher port
            no_filter (bool): Disable EMG filtering
        
        Returns:
            bool: True if process started successfully, False otherwise
        """
        if self.is_running:
            self.status_changed.emit("Bridge already running")
            return True
            
        # Build command with arguments
        cmd = [sys.executable, self.bridge_script_path]
        if simulate:
            cmd.append("--simulate")
        cmd.extend(["--device", device])
        cmd.extend(["--channels", channels])
        cmd.extend(["--rate", str(rate)])
        cmd.extend(["--port", str(port)])
        if no_filter:
            cmd.append("--no-filter")
            
        # Start the bridge process
        try:
            # Use subprocess.Popen to start the process
            self.bridge_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Start a thread to monitor the process output
            self.monitor_thread = threading.Thread(
                target=self._monitor_process,
                daemon=True
            )
            self.monitor_thread.start()
            
            # Wait a moment to check if process started successfully
            time.sleep(0.5)
            if self.bridge_process.poll() is not None:
                # Process exited immediately
                return_code = self.bridge_process.poll()
                stderr = self.bridge_process.stderr.read()
                self.status_changed.emit(f"Failed to start bridge: Exit code {return_code}\n{stderr}")
                self.bridge_process = None
                return False
                
            self.is_running = True
            self.status_changed.emit("Bridge started successfully")
            return True
            
        except Exception as e:
            self.status_changed.emit(f"Error starting bridge: {e}")
            if self.bridge_process:
                try:
                    self.bridge_process.terminate()
                except:
                    pass
                self.bridge_process = None
            return False
            
    def stop_bridge(self):
        """Stop the DAQ-to-ZMQ bridge process"""
        if not self.is_running or self.bridge_process is None:
            self.status_changed.emit("Bridge not running")
            return True
            
        try:
            # Send SIGTERM signal to gracefully terminate the process
            if sys.platform == 'win32':
                self.bridge_process.terminate()
            else:
                os.kill(self.bridge_process.pid, signal.SIGTERM)
                
            # Wait for process to terminate
            for _ in range(10):
                if self.bridge_process.poll() is not None:
                    break
                time.sleep(0.1)
                
            # Force kill if not terminated
            if self.bridge_process.poll() is None:
                if sys.platform == 'win32':
                    self.bridge_process.kill()
                else:
                    os.kill(self.bridge_process.pid, signal.SIGKILL)
                    
            self.is_running = False
            self.bridge_process = None
            self.status_changed.emit("Bridge stopped")
            return True
            
        except Exception as e:
            self.status_changed.emit(f"Error stopping bridge: {e}")
            return False
            
    def _monitor_process(self):
        """Monitor the bridge process output"""
        try:
            # Read process output line by line
            for line in iter(self.bridge_process.stdout.readline, ''):
                if line:
                    # Forward important messages to UI
                    line = line.strip()
                    if "Error" in line or "WARNING" in line or "DAQ" in line:
                        self.status_changed.emit(line)
                else:
                    break
                    
            # Process has ended
            if self.bridge_process:
                return_code = self.bridge_process.poll()
                if return_code is not None and return_code != 0:
                    stderr = self.bridge_process.stderr.read()
                    self.status_changed.emit(f"Bridge exited with code {return_code}\n{stderr}")
                self.is_running = False
                
        except Exception as e:
            self.status_changed.emit(f"Error monitoring bridge: {e}")
            self.is_running = False