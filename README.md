# Speech Minutes - 演讲纪要

听演讲时，一个可以实时将语音转换为文字记录、语言翻译的工具

## Installation

1) clone:

```bash

git clone https://github.com/harryzhu/speech-minutes.git

cd speech-minutes

```


2) 下载 模型 放在 models 文件夹下

  * [Qwen3-ASR-1.7B](https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)

  * [Qwen3.5-2B-MLX-8bit](https://www.modelscope.cn/models/mlx-community/Qwen3.5-2B-MLX-8bit)

  * [UVR](https://huggingface.co/hazhu/mlx-uvr-mdx-net)


## Usage

1) 开启一个新的终端窗口，执行 `recording_sentence.py` 监听麦克风，实时听取演讲者的语音

```bash

python recording_sentence.py

```

2) 开启另一个新的终端窗口，执行 `speech2text.py` ，会将麦克风中的语音转为文字，

   此处使用 `Qwen3-ASR` 本地部署完成，无需调用任何云端 API

```python

python speech2text.py

```

## Optional - 可选功能

3) 开启一个新的终端窗口，执行 `text2translation.py`， 可以生成翻译：中译英 或者 英译中.

此功能需要本地部署 `Qwen3.5-2B-MLX-8bit` ，仅在高内存机器上运行较好。

```bash

python text2translation.py

```

## Optimization

1) 先用 `realtime_dbfs.py` 检测演讲者现场的声音分贝，
按显示的数值预估修改 `config.py` 中的下面者两项值，可以优化语音的自动断句

```python
# 分贝值低于 -50 视为静音，
# 高于 -27 视为有人在讲话
rec_silence_threshold_dbps = -50.0
rec_speech_threshold_dbps = -27.0
```

2) 在另外一台机器上运行 `recording_single.py` ，可以一直记录演讲者的声音，中途不会自动截断语句，便于以后重听、核对。

```bash

python recording_single.py

```

## 自动字幕

1) 在 `srt/` 目录下会自动生成 字幕文件： `a.srt`






