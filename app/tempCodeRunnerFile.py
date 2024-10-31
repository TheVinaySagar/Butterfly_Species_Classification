import sys
import os.path

# Add the parent directory to the system path for module imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)