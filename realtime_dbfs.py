import os
import time
import math
import queue

from config import *
from utils import *

q_test_dbfs = queue.Queue()

def callback_test_dbfs(indata, frames, time, status):
    if status:
        print(status)
    q_test_dbfs.put(indata.copy())

def recording_test_dbfs():
    with sd.InputStream(samplerate=rec_samplerate,
                        channels=rec_channel, 
                        blocksize=rec_samplerate,
                        callback=callback_test_dbfs):
        print('press Ctrl+C to stop the recording')
        print('#' * 80)
        while True:
            dBFS = get_dBFS(q_test_dbfs.get())
            print(f'{dBFS:.2f}')

if __name__ == "__main__":
    print("play the audio, check the dBFS, then update config.py: ")
    recording_test_dbfs()
    


