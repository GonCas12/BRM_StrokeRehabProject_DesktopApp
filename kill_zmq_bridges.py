#!/usr/bin/env python3
"""
Utility script to forcibly kill any running DAQ-to-ZMQ bridge processes
"""

import os
import sys
import signal
import subprocess
import time

def kill_bridge_processes():
    """Find and kill any running bridge processes"""
    killed_count = 0
    
    try:
        # For Windows
        if sys.platform.startswith('win'):
            # Find Python processes that are running run_mock_daq_test.py
            cmd = 'wmic process where "commandline like \'%run_mock_daq_test.py%\'" get processid'
            result = subprocess.check_output(cmd, shell=True).decode('utf-8')
            
            # Extract process IDs
            for line in result.strip().split('\n')[1:]:  # Skip header line
                if line.strip():
                    try:
                        pid = int(line.strip())
                        os.kill(pid, signal.SIGTERM)
                        print(f"Terminated process {pid}")
                        killed_count += 1
                    except (ValueError, ProcessLookupError) as e:
                        print(f"Error terminating process: {e}")
        
        # For Unix/Linux/Mac
        else:
            # Find Python processes that are running run_mock_daq_test.py
            cmd = "ps -ef | grep run_mock_daq_test.py | grep -v grep | awk '{print $2}'"
            result = subprocess.check_output(cmd, shell=True).decode('utf-8')
            
            # Extract process IDs
            for line in result.strip().split('\n'):
                if line.strip():
                    try:
                        pid = int(line.strip())
                        os.kill(pid, signal.SIGTERM)
                        print(f"Terminated process {pid}")
                        killed_count += 1
                    except (ValueError, ProcessLookupError) as e:
                        print(f"Error terminating process: {e}")
    
    except Exception as e:
        print(f"Error finding/killing processes: {e}")
    
    return killed_count

if __name__ == "__main__":
    killed = kill_bridge_processes()
    print(f"Killed {killed} bridge processes")
    
    # Force free the ZMQ port with netstat + taskkill on Windows
    if sys.platform.startswith('win'):
        try:
            cmd = 'netstat -ano | findstr "5555"'
            result = subprocess.check_output(cmd, shell=True).decode('utf-8')
            
            for line in result.strip().split('\n'):
                if "LISTENING" in line and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                            print(f"Killed process {pid} using port 5555")
                        except Exception as e:
                            print(f"Error killing process {pid}: {e}")
        except Exception as e:
            print(f"Error freeing port: {e}")
    
    # Give processes time to actually close
    time.sleep(1)