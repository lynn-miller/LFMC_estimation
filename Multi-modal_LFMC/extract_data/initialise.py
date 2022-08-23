"""Initialise notebook or script"""

import sys
import os
import pathlib

# Add main code directory to path
common_dir = pathlib.Path(__file__).parent.parent
code_dir = common_dir.parent
sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(common_dir))

# Set Tensorflow message level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"