import os
import time
import math

import re
from io import BytesIO
from glob import glob

import numpy as np
import sounddevice as sd
import soundfile as sf
from config import *

def get_dBFS(indata):
    rms_per_channel = np.sqrt(np.mean(indata**2, axis=0))
    rms = np.max(rms_per_channel)
    dBFS = 20 * np.log10(rms) if rms > 1e-10 else -np.inf
    return dBFS


     
