import os
import time
import queue
import signal
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import numpy as np
import sounddevice as sd
import soundfile as sf
from config import *
from utils import *

#
q_allinone = queue.Queue()
#
sd.default.samplerate = rec_samplerate

def callback_allinone(indata, frames, time, status):
    if status:
        print(status)
    q_allinone.put(indata.copy())

def recording_allinone():
    with sf.SoundFile("one/single.wav", mode='x', samplerate=rec_samplerate,
                        channels=1) as fw:
        with sd.InputStream(samplerate=rec_samplerate,
                            channels=rec_channel, 
                            blocksize=rec_samplerate * 3,
                            callback=callback_allinone):
            print('press Ctrl+C to stop the recording')
            print('#' * 20)
            while True:
                fw.write(q_allinone.get())

if __name__ == "__main__":
    if not os.path.isdir("one"):
            os.makedirs("one", exist_ok=True)
    
    if os.path.isfile("one/single.wav"):
        os.remove("one/single.wav")  

    recording_allinone()
    
   


