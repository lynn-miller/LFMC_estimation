"""Initialise notebook or script"""

import sys
import os
# import platform

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, code_dir)

# Set Tensorflow message level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"