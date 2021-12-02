"""Initialise notebook or script"""

import sys
import os
import platform

# Add main code directory to path
if platform.system() == 'Windows':
    code_dir = os.path.dirname(sys.path[0])
else:
    code_dir = '..'
sys.path.insert(0, code_dir)

# Set Tensorflow message level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"