import soundfile as sf
import os 
import librosa
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import onnxruntime as ort
from tqdm import tqdm

def mx_hann_periodic(M: int):
    n = mx.arange(M, dtype=mx.float32)
    return 0.5 - 0.5 * mx.cos(2 * mx.pi * n / M)

def mx_view_as_real(z: mx.array):
    re = mx.real(z)
    im = mx.imag(z)
    return mx.stack([re, im], axis=-1)

def mx_view_as_complex(ri: mx.array):
    re = ri[..., 0]
    im = ri[..., 1]
    return re + 1j * im

class ConvTDFNet(nn.Module):
    def __init__(self, target_name, L, dim_f, dim_t, n_fft, hop=1024):
        super(ConvTDFNet, self).__init__()
        self.dim_c = 4
        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = mx_hann_periodic(M=self.n_fft)
        self.target_name = target_name
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        self.freq_pad = mx.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t])
        self.n = L // 2

    def stft(self, x):
        x = mx.array(x.reshape([-1, self.chunk_size]))
        x = mx.fft.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            pad_mode='reflect',
            center=True,
            onesided=True,
            )
        x = mx_view_as_real(x)
        x = x.transpose((0, 3, 1, 2))
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t])
        x = x.reshape([-1, self.dim_c, self.n_bins, self.dim_t])
        return x[:, :, : self.dim_f]

    # Inversed Short-time Fourier transform (STFT).
    def istft(self, x, freq_pad=None):
        if freq_pad is None:
            freq_pad = mx.tile(self.freq_pad,(x.shape[0], 1, 1, 1))
        
        x = mx.concatenate([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t])
        x = x.reshape([-1, 2, self.n_bins, self.dim_t])
        x = x.transpose([0, 2, 3, 1])
        x = mx_view_as_complex(x)
        
        x = mx.fft.istft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop, 
            window=self.window, 
            center=True,
            onesided=False,
        )
        return x.reshape([-1, c, self.chunk_size])

class Predictor:
    def __init__(self, model_path, dim_f, dim_t, n_fft, denoise=True, margin=44100, chunks=15):
        self.model_ = ConvTDFNet(
            target_name="vocals",
            L=11,
            dim_f=dim_f,
            dim_t=dim_t,
            n_fft=n_fft
        )
        self.denoise = denoise
        self.model_path = model_path
        self.margin = margin
        self.chunks = chunks
        self.model = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])        
        
    def demix(self, mix):
        samples = mix.shape[-1]
        margin = self.margin
        chunk_size = self.chunks * 44_100
        
        assert not margin == 0, "margin cannot be zero!"
        
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        
        for skip in range(0, samples, chunk_size):
            counter += 1
            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)
            start = skip - s_margin
            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break
            
        sources = self.demix_base(segmented_mix, margin_size=margin)
        return sources

    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes))
        progress_bar.set_description("Processing")    

        for mix in mixes:
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = mx.concatenate(
                (mx.zeros((2, trim)), 
                 mx.array(cmix), 
                 mx.zeros((2, pad)), 
                 mx.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = mix_p[:, i : i + model.chunk_size]
                mix_waves.append(waves)
                i += gen_size
            
            mix_waves = mx.array(mix_waves, dtype=mx.float32)
            #mix_waves = mx.stack(mix_waves, axis=0)
            
            _ort = self.model
            spek = model.stft(mix_waves)
            #print(type(spek))
            spek_np = np.array(spek)
            # t1 = time.time()
            if self.denoise:             
                spec_pred = (
                    - _ort.run(None, {"input": -spek_np})[0] * 0.5
                    + _ort.run(None, {"input": spek_np})[0] * 0.5
                )
                tar_waves = model.istft(mx.array(spec_pred))
            else:
                tar_waves = model.istft(
                    mx.array(_ort.run(None, {"input": spek_np})[0])
                )
            # print(time.time() - t1)
            tar_signal = (
                tar_waves[:, :, trim:-trim]
                .transpose((1,0,2))
                .reshape(2, -1))[:, :-pad]

            start = 0 if mix == 0 else margin_size
            end = None if mix == list(mixes.keys())[::-1][0] else -margin_size

            if margin_size == 0:
                end = None
            
            sources.append(tar_signal[:, start:end])
            progress_bar.update(1)

            chunked_sources.append(sources)
        _sources = mx.concatenate(mx.array(chunked_sources), axis=-1)

        progress_bar.close()
        return _sources

    def predict(self, file_path):
        mix, rate = librosa.load(file_path, mono=False, sr=44100)

        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
            
        mix = mix.T  
        sources = self.demix(mix.T)
        opt = sources[0].T

        return (mix - opt, opt, rate)





