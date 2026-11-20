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
q = queue.Queue()
sd.default.samplerate = rec_samplerate
#
is_stop_all = False

async def recording(rec_buffer, seg_id):
    def callback(indata, frames, time_info, status):
        nonlocal idx
        nonlocal rec_buffer
        nonlocal len_rec_buffer
        nonlocal silence_count
        if status:
            print(status)
        remainder = len_rec_buffer - idx
   
        if remainder == 0 or is_stop_all == True:
            loop.call_soon_threadsafe(evt.set)
            raise sd.CallbackStop()

        if remainder != 0 and remainder < rec_samplerate*7:  
            dBFS = get_dBFS(indata)   
            if dBFS < rec_silence_threshold_dbps: 
                silence_count += 1
                if silence_count >= mic_silence_count:
                    loop.call_soon_threadsafe(evt.set) 
                    raise sd.CallbackStop() 
            
            if dBFS > float(rec_speech_threshold_dbps):  
                silence_count = 0                   
            #print("dBFS:",dBFS, ", silence_count: ", silence_count)                
        #
        write_len = min(frames, remainder)
        rec_buffer[idx:idx+len(indata)] = indata[:write_len]
        idx += len(indata)
         

    loop = asyncio.get_running_loop()
    evt = asyncio.Event()
    idx = 0
    silence_count = 0
    t_start = time.perf_counter()
    len_rec_buffer = len(rec_buffer)
    try:
        with sd.InputStream(callback=callback,
                            samplerate=rec_samplerate,
                            dtype=rec_buffer.dtype,
                            channels=rec_buffer.shape[1],
                            blocksize=rec_blocksize,
                            latency='low'):
            #print(f"## task_producer: {seg_id} ##")      
            await evt.wait()    
            t_stop = time.perf_counter()
            #
            f_t_duration = round(t_stop - t_start)
            rec_buffer = rec_buffer[:idx] 
            out_file = f"wav/_src/a_{seg_id}_({f_t_duration})_[{t_start:.3f}-{t_stop:.3f}].wav"
            save_audio = np.concatenate([audio_empty, rec_buffer, audio_empty], axis=0)   
            q.put((out_file, save_audio))
            # reset
            idx = 0
            silence_count = 0
    except Exception as err:
        print(err)
        return None


async def task_producer():
    with open("wav/_src/ts_begin.txt", "w")as fw:
        ts = time.perf_counter()
        fw.write(f'{ts:.3f}')
    
    rec_buffer = np.zeros((rec_max_frames, rec_channel), dtype='float32')
    seg_id = 0
    while True:
        if is_stop_all:
            break
        await recording(rec_buffer, seg_id)
        seg_id += 1

def file_saver():
    while True:
        if is_stop_all:
            break
        try:
            if q.qsize() > 0:
                item = q.get()
                sf.write(item[0], item[1], rec_samplerate)
            else:
                time.sleep(0.3)
        except Exception as err:
            print(err)
            time.sleep(0.3)

def mic_listen():
    asyncio.run(task_producer())

def terminate_all(signum, frame):
    global is_stop_all
    is_stop_all = True
    print(f" Bye ...")
    time.sleep(1)
    os._exit(0)    


if __name__ == "__main__":
    signal.signal(signal.SIGINT, terminate_all)

    if not os.path.isdir("wav/_src"):
        os.makedirs("wav/_src", exist_ok=True) 
     
    default_input_device = sd.query_devices(kind='input')
    if "default_samplerate" in default_input_device.keys():
        if default_input_device["default_samplerate"] != rec_samplerate:
            print("device default_samplerate: ", default_input_device["default_samplerate"])
            print("rec_samplerate: ",rec_samplerate)
            print(f'pls set rec_samplerate = {default_input_device["default_samplerate"]}')

    with ThreadPoolExecutor(4) as executor:
        t1 = executor.submit(mic_listen)
        t2 = executor.submit(file_saver)
        tasks = [t1, t2]
        wait(tasks, return_when=ALL_COMPLETED)
        for task in tasks:
            print(task.result())
    
   


