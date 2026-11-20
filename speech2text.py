import os
import time
import math
from mlx_audio.stt.utils import load_model
from mlx_audio.stt.generate import generate_transcription

import re
from glob import glob

import numpy as np
import sounddevice as sd
import soundfile as sf
from config import *

model_path_asr = "models/Qwen3-ASR-1.7B-8bit"
if not os.path.isdir(model_path_asr):
    model_path_asr = '/Volumes/HDD4/_models/Qwen3/Qwen3-ASR-1.7B'

model_asr = load_model(model_path_asr)
ts_begin_file = "wav/_src/ts_begin.txt"
ts_begin = 0.0


def sec2hms(seconds=0):
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    ms = str(round(seconds - math.floor(seconds),3))[2:]
    return f"{int(hours):02d}:{int(minutes):02d}:{int(sec):02d},{ms}"

def wav2txt(fpath=""):
    transcription = generate_transcription(
        model=model_asr,
        audio=fpath,
        output_path=f'txt/{os.path.basename(fpath)}',
        format="txt",
        language=speech_language,
        verbose=False,
    )

    #print(tgt_trans.text)
    t_fm = 0.0
    t_to = 0.0
    ts_pattern = r'\[\d+\.\d+\-\d+\.\d+\]'
    matches = re.findall(ts_pattern, fpath)
    if len(matches) == 1:
        b_e = matches[0][1:-1]
        fm_to = b_e.split("-")
        if len(fm_to) == 2:
            t_fm = float(fm_to[0])
            t_to = float(fm_to[1])
    if ts_begin > 0.0:
        t_fm = t_fm - ts_begin
        t_to = t_to - ts_begin
    #idx = os.path.basename(fpath)[0:os.path.basename(fpath).index("_",2)]
    subtitle = f"{sec2hms(t_fm)} --> {sec2hms(t_to)} \n{transcription.text}\n"
    #print(subtitle)
    with open("srt/a.srt","a+")as fw:
        subtitle = subtitle.replace("，"," ").replace("？"," ").replace("！"," ").replace("。"," ")
        print(subtitle)    
        fw.write(f"{subtitle}\n")
    return None

def task_consumer():
    print("## speech to text ##")
    seg_id = 0
    fpath = ""
    while True:
        flist = glob(f'wav/_src/a_{seg_id}_*.wav')
        #print(flist)
        if len(flist) != 0:
            fpath = flist[0]
            fvocal = fpath.replace("wav/_src/", "wav/vocal/")
            if os.path.exists(fvocal):
                fpath = fvocal
            # 等待0.3秒，避免ASR输出大量“嗯”
            time.sleep(0.3)
            wav2txt(fpath)
            seg_id += 1


if __name__ == "__main__":
    if os.path.exists(ts_begin_file):
        with open(ts_begin_file, 'r')as fr:
            ts_begin = float(fr.read())
    print(f"ts_begin: {ts_begin}")

    if not os.path.isdir("srt"):
        os.makedirs("srt", exist_ok=True)
    if not os.path.isdir("txt"):
        os.makedirs("txt", exist_ok=True)

    if os.path.exists("srt/a.srt"):
        os.remove("srt/a.srt")

    task_consumer()

    
    
   


