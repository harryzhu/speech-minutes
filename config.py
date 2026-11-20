import numpy as np

speech_language = 'zh'
translation_language = 'en'
#
rec_samplerate = 48_000
#
# 100ms 一块
rec_blocksize = int(rec_samplerate*0.1) 
#
rec_max_duration = 12
rec_max_frames = rec_max_duration * rec_samplerate
#
# 低于 -50.0 视为静音，高于 -27.0 视为有人在讲话
rec_silence_threshold_dbps = -50.0
rec_speech_threshold_dbps = -27.0
rec_channel = 1
# 说话结束 300ms 后，如果没有声音就截断, 0.1*3=0.3秒
mic_silence_count = 3 
# 在头尾添加 500ms 空白，ASR识别效果更好
audio_empty = np.zeros((rec_blocksize*5, rec_channel), dtype='float32')

