import os
from glob import glob
from config import *
from utils import *
from uvr import *


def extract_vocals(fpath):
    music, vocal = separate_audio(fpath, output_dir="wav")
    return music, vocal

if __name__ == "__main__":
    files = glob(f'wav/_src/a_*.wav')
    for f in files:
        extract_vocals(f)

    
   


