"""Initialise notebook or script"""

import sys
import os
import pathlib

# Add main code directory to path
code_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

# Set Tensorflow message level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"