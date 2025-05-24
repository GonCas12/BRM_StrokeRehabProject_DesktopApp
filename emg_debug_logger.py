#!/usr/bin/env python3
"""
EMG Classifier Debug Logger
"""
import os
import time
import sys
from datetime import datetime

class EMGLogger:
    """Simple file-based logger with timestamps"""
    
    def __init__(self, filename="emg_classifier_debug.log", console=True):
        self.filename = filename
        self.console = console
        
        # Create or truncate the log file
        with open(self.filename, 'w') as f:
            f.write(f"=== EMG Classifier Debug Log ===\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log(self, message, level="INFO"):
        """Write a message to the log with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]  # Include milliseconds
        formatted = f"[{timestamp}] {level}: {message}"
        
        # Write to file
        with open(self.filename, 'a') as f:
            f.write(formatted + '\n')
            f.flush()  # Ensure it's written immediately
        
        # Also print to console if enabled
        if self.console:
            print(formatted, flush=True)
    
    def info(self, message):
        """Log an info message"""
        self.log(message, "INFO")
    
    def debug(self, message):
        """Log a debug message"""
        self.log(message, "DEBUG")
    
    def warning(self, message):
        """Log a warning message"""
        self.log(message, "WARNING")
    
    def error(self, message):
        """Log an error message"""
        self.log(message, "ERROR")

# Global logger instance
logger = EMGLogger()

# Replace print with logger.info
def log_info(message):
    logger.info(message)

def log_debug(message):
    logger.debug(message)

def log_warning(message):
    logger.warning(message)

def log_error(message):
    logger.error(message)