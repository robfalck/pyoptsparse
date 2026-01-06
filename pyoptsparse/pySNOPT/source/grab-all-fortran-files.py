#!/usr/bin/env python

# Standard Python modules
import os
from pathlib import Path

CURDIR = os.path.abspath(os.path.dirname(__file__))

TO_SKIP = [
    "sn27lu77.f",  # Old Fortran 77 LU version (exclude, use sn27lu90.f instead)
    "snopth.f",     # Exclude this file
]

for path in Path(CURDIR).glob("*.f"):
    if os.path.basename(path) not in TO_SKIP:
        print(path)
