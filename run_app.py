import sys
from PyQt5.QtWidgets import QApplication

# Initialize QApplication first
app = QApplication(sys.argv)

# Make app available as a global to imported modules
import builtins
builtins.qapp = app

# Now it's safe to import your main module
from main_v1 import run_application

if __name__ == "__main__":
    # Run the application with the existing QApplication instance
    sys.exit(run_application(app))