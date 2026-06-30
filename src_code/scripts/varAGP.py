"""
Variational calculation of AGP for OQS
================================

Script for variational calculation of the adiabatic gauge potential (AGP) 
for open quantum systems (OQS) using the Lindblad master equation framework.
"""

# Import necessary libraries and main.py from src folder

import os
import sys
import pickle
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import matplotlib.transforms as mtransforms
import matplotlib.colors as mcolors

# Move into repo/src to import Lindblad library
p = Path.cwd()
os.chdir(p)
p = p.parent / "src"   # go up one level, then into src
print(p)
os.chdir(p)

import sys
sys.path.append(str(p))
from main import *   # Import the library for Walsh functions

