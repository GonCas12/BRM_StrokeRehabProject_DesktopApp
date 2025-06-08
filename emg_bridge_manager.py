import subprocess
import os
import signal
import time
import threading
import sys
from PyQt5.QtCore import QObject, pyqtSignal

class EMGBridgeManager(QObject):
    """
    Manages the external DAQ-to-ZMQ bridge process without blocking the UI
    """
    status_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.bridge_process = None
        self.is_running = False
        self.process_output_thread = None
        self.terminate_flag = False
    
    def start_bridge(self, simulate=True, test_file=None):
        """Start the DAQ-to-ZMQ bridge process with optional test file"""
        if self.is_running:
            self.status_changed.emit("Already running")
            return True
            
        try:
            # First kill any existing processes
            print("Checking for existing bridge processes...")
            try:
                subprocess.run([sys.executable, 'kill_zmq_bridges.py'], 
                            timeout=5, 
                            check=False,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
            except Exception as e:
                print(f"Error running cleanup script: {e}")
            
            # Give the system time to fully release resources
            time.sleep(1)
            
            # Now start the bridge
            cmd = [sys.executable, 'run_mock_daq_test.py']
            
            # Add mock flag if simulating
            if simulate:
                cmd.append('--mock')
                
            # Add test file if provided
            if test_file:
                cmd.extend(['--file', test_file])
            
            # Start the process
            print(f"Starting bridge with command: {' '.join(cmd)}")
            self.bridge_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Start thread to handle output without blocking
            self.terminate_flag = False
            self.process_output_thread = threading.Thread(
                target=self._monitor_process_output,
                daemon=True
            )
            self.process_output_thread.start()
            
            # Give it a moment to start up
            time.sleep(1)
            
            if self.bridge_process.poll() is None:
                self.is_running = True
                self.status_changed.emit("Running")
                return True
            else:
                self.status_changed.emit("Failed to start")
                return False
                
        except Exception as e:
            self.status_changed.emit(f"Error: {str(e)}")
            return False
    
    def _monitor_process_output(self):
        """Monitor process output in a separate thread"""
        while self.bridge_process and not self.terminate_flag:
            if self.bridge_process.poll() is not None:
                # Process has terminated
                if not self.terminate_flag:  # Only emit if not manually terminated
                    self.is_running = False
                    self.status_changed.emit("Terminated unexpectedly")
                break
                
            # Read output without blocking
            try:
                output_line = self.bridge_process.stdout.readline()
                if output_line:
                    print(f"Bridge output: {output_line.strip()}")
                else:
                    # No more output but process still running
                    time.sleep(0.1)
            except:
                time.sleep(0.1)
                
        print("Process output monitoring stopped")
    
    def stop_bridge(self):
        """Stop the DAQ-to-ZMQ bridge process"""
        if not self.is_running or not self.bridge_process:
            self.status_changed.emit("Not running")
            return True
        
        try:
            # Signal thread to stop
            self.terminate_flag = True
            
            # Terminate process
            print("Stopping EMG bridge...")
            self.bridge_process.terminate()
            
            # Give it a moment to terminate
            for _ in range(10):  # Wait up to 1 second
                if self.bridge_process.poll() is not None:
                    break
                time.sleep(0.1)
            
            # Force kill if still running
            if self.bridge_process.poll() is None:
                print("Bridge didn't terminate gracefully, forcing kill...")
                if os.name == 'nt':  # Windows
                    os.kill(self.bridge_process.pid, signal.CTRL_BREAK_EVENT)
                else:  # Unix/Linux
                    os.kill(self.bridge_process.pid, signal.SIGKILL)
            
            # Wait for monitoring thread to finish
            if self.process_output_thread and self.process_output_thread.is_alive():
                self.process_output_thread.join(timeout=2.0)
                
            self.bridge_process = None
            self.is_running = False
            self.status_changed.emit("Stopped")
            return True
            
        except Exception as e:
            self.status_changed.emit(f"Error stopping: {str(e)}")
            return False