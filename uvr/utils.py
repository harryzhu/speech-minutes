import os 
import soundfile as sf
from .predictor import *

model_path_hq5 = f'{os.path.dirname(__file__)}/../models/UVR/UVR-MDX-NET-Inst_HQ_5.onnx'
if not os.path.isdir(model_path_hq5):
    model_path_hq5 = "/Volumes/HDD4/_models/UVR/UVR-MDX-NET-Inst_HQ_5.onnx"

model_path_hq3 = f'{os.path.dirname(__file__)}/../models/UVR/UVR-MDX-NET-Inst_HQ_3.onnx'
if not os.path.isdir(model_path_hq3):
    model_path_hq3 = "/Volumes/HDD4/_models/UVR/UVR-MDX-NET-Inst_HQ_3.onnx"

predictor = Predictor(
        model_path=model_path_hq5,
        dim_f=2560,
        dim_t=8,
        n_fft=6144,
        #denoise=True,
        denoise=False,
        margin=44100,
        chunks=15
    )
# predictor = Predictor(
#         model_path=model_path_hq3,
#         dim_f=3072,
#         dim_t=8,
#         n_fft=6144,
#         denoise=True,
#         # denoise=False,
#         margin=44100,
#         chunks=15
#     )
def separate_audio(file_path, output_dir):
    #print(ort.get_available_providers())
    if not os.path.isdir(f"{output_dir}/music"):
        os.makedirs(f"{output_dir}/music", exist_ok=True)
    if not os.path.isdir(f"{output_dir}/vocal"):
        os.makedirs(f"{output_dir}/vocal", exist_ok=True)
    vocals, no_vocals, sampling_rate = predictor.predict(file_path)
    filename = os.path.splitext(os.path.split(file_path)[-1])[0]
    f_music = f"{output_dir}/music/{filename}.wav"
    f_vocal = f"{output_dir}/vocal/{filename}.wav"
    sf.write(f_music, no_vocals, 44_100)
    sf.write(f_vocal, vocals, 44_100)

    return f_music, f_vocal


   