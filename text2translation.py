import os
import time
import math
from mlx_audio.stt.utils import load_model
from mlx_audio.stt.generate import generate_transcription

from mlx_vlm import load, generate

import re
from io import BytesIO
from glob import glob

import numpy as np
import sounddevice as sd
import soundfile as sf
from config import *

model_path_trans = "models/Qwen3.5-2B-MLX-8bit"
if not os.path.isdir(model_path_trans):
    model_path_trans = "/Volumes/HDD4/_models/Qwen3/Qwen3.5-2B-MLX-8bit"

ts_begin_file = "wav/_src/ts_begin.txt"
ts_begin = 0.0

model_trans, processor_trans = load(model_path_trans)


def sec2hms(seconds=0):
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    ms = str(round(seconds - math.floor(seconds),3))[2:]
    return f"{int(hours):02d}:{int(minutes):02d}:{int(sec):02d},{ms}"

def txt2translation(fpath=""):
    ftext = ""
    with open(fpath, "r")as fr:
        ftext = fr.read().strip()
        fr.close()
    if len(ftext) == 0:
        return None
    trans_text = ''
    pt_content = f"不要添加额外描述，将下面的中文翻译成英文: {ftext}"
    if speech_language == 'en':
        pt_content = f"do not add extra words, just translate the following English sentences into Chinese: {ftext}"
    prompt_trans = processor_trans.apply_chat_template(
            [{"role": "user", 
                "content": pt_content}],
            add_generation_prompt=True,
        )
    tgt_trans = generate(
        model = model_trans,
        processor = processor_trans,
        prompt=prompt_trans,
        max_tokens=200
    )
    trans_text = tgt_trans.text
    with open(fpath, 'a+')as fw:
        fw.write(f"\n\n{trans_text}")
        fw.close()

    
    #print(tgt_trans.text)
    #audio = AudioSegment.from_wav(fpath)
    #print(f"{round(len(audio)/1000)}s: {transcription.text}")
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
    subtitle = f"{sec2hms(t_fm)} --> {sec2hms(t_to)} \n{ftext}\n{trans_text}"
    #print(subtitle)
    with open("srt/a.srt","a+")as fw:
        subtitle = subtitle.replace("，"," ").replace("？"," ").replace("！"," ").replace("。"," ")
        print(subtitle)    
        fw.write(f"{subtitle}\n")
    return None

def task_consumer():
    print("## task_consumer ##")
    seg_id = 0
    fpath = ""
    while True:
        flist = glob(f'txt/a_{seg_id}_*.txt')
        #print(flist)
        if len(flist) != 0:
            fpath = flist[0]
            # 等待0.3秒
            time.sleep(0.3)
            txt2translation(fpath)
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
   


