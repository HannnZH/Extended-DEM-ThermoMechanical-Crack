"""
Shared preamble for the example scripts.

Each example is a standalone script that can be run from anywhere, so it puts the repository
root on the import path before importing the package.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
