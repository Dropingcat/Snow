import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal, ndimage
from scipy.signal import find_peaks, butter, filtfilt, savgol_filter
from scipy import interpolate, stats
import librosa
import os
import json
from datetime import datetime
import warnings
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import logging

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  sounddevice не установлен. Только сохранение в WAV.")

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ========================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ========================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('SnowSimulator')
logger.setLevel(logging.INFO)

# ========================
# КЭШИРОВАНИЕ
# ========================

def load_cached_params(cache_file='ref_params_v16.json'):
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            print(f"✅ Загружено из кэша: {cache_file}")
            if 'f_spectrum' in params and isinstance(params['f_spectrum'], list):
                params['f_spectrum'] = np.array(params['f_spectrum'])
            if 'Pxx_spectrum' in params and isinstance(params['Pxx_spectrum'], list):
                params['Pxx_spectrum'] = np.array(params['Pxx_spectrum'])
            if 'amp_envelope' in params and isinstance(params['amp_envelope'], list):
                params['amp_envelope'] = np.array(params['amp_envelope'])
            params['success'] = True
            return params
        except Exception as e:
            logger.warning(f"⚠️  Ошибка загрузки кэша: {e}")
    return None

def save_cached_params(params, cache_file='ref_params_v16.json'):
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return obj
    
    params_to_save = {k: convert_for_json(v) for k, v in params.items() if k != 'audio'}
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(params_to_save, f, indent=2, ensure_ascii=False)
    print(f"💾 Сохранено в кэш: {cache_file}")


# ========================
# АНАЛИЗ ЭТАЛОНА V16
# ========================

def analyze_reference_enhanced(filename, num_textures=8, max_duration=60):
    default_params = {
        'success': False,
        'sr': 44100,
        'step_frequency': 1.8,
        'contact_duration': 0.3,
        'peak_rms_ratio': 28.0,
        'high_freq_cutoff': 6500,
        'textures': [],
        'spectral_dynamics': {'centroid_range': [1000, 4000], 'kurtosis_stats': [2, 3, 4]}
    }
    
    try:
        audio, sr = librosa.load(filename, sr=None, mono=True, duration=max_duration)
        print(f"✅ Загружено: {len(audio)} сэмплов ({len(audio)/sr:.1f} с), {sr} Гц")
    except Exception as e:
        logger.warning(f"⚠️  librosa не удалось: {e}")
        try:
            sr, audio = wavfile.read(filename)
            if audio.ndim > 1:
                audio = audio[:, 0]
            audio = audio.astype(np.float64) / 32768.0
            if max_duration > 0:
                audio = audio[:int(max_duration * sr)]
            print(f"✅ Загружено через wavfile: {len(audio)} сэмплов, {sr} Гц")
        except Exception as e2:
            logger.error(f"❌ Не удалось загрузить файл: {e2}")
            return default_params
    
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    duration = len(audio) / sr
    
    energy = librosa.feature.rms(y=audio, frame_length=int(sr*0.02), hop_length=int(sr*0.005))[0]
    time_axis = np.arange(len(energy)) * 0.005
    
    peaks, props = find_peaks(energy, height=0.15*np.max(energy), 
                              distance=int(0.2/0.005), prominence=0.05)
    
    if len(peaks) < 3:
        logger.warning("⚠️  Мало шагов для анализа")
        return default_params
    
    print(f"👣 Найдено {len(peaks)} шагов")
    
    if len(peaks) >= 2:
        step_intervals = np.diff(time_axis[peaks])
        step_frequency = 1 / np.mean(step_intervals)
    else:
        step_frequency = 1.8
    
    contact_durations = []
    for peak_idx in peaks[:min(5, len(peaks))]:
        segment = energy[max(0, peak_idx-20):min(len(energy), peak_idx+80)]
        above = np.where(segment > 0.1*np.max(segment))[0]
        if len(above) > 1:
            contact_durations.append((above[-1] - above[0]) * 0.005)
    
    contact_duration = np.median(contact_durations) if contact_durations else 0.3
    print(f"⏱️  Контакт: {contact_duration*1000:.0f} мс")
    
    textures = []
    min_len = int(0.25 * sr)
    
    for i, peak_idx in enumerate(peaks[:min(num_textures, len(peaks))]):
        t_start = max(0, time_axis[peak_idx] - 0.05)
        t_end = min(duration, t_start + contact_duration + 0.1)
        
        start_sample = int(t_start * sr)
        end_sample = int(t_end * sr)
        window = audio[start_sample:end_sample]
        
        if len(window) < min_len:
            window = np.pad(window, (0, min_len - len(window)), mode='constant')
        elif len(window) > int(0.45 * sr):
            window = window[:int(0.45 * sr)]
        
        hop = int(sr * 0.01)
        centroids = []
        kurtoses = []
        
        for t in range(0, len(window)-2048, hop):
            spec = np.abs(np.fft.rfft(window[t:t+2048]))
            f = np.fft.rfftfreq(2048, 1/sr)
            if np.sum(spec) > 1e-10:
                centroid = np.sum(f * spec) / np.sum(spec)
                kurt = stats.kurtosis(spec) if len(spec) > 4 else 3.0
                centroids.append(centroid)
                kurtoses.append(kurt)
        
        textures.append({
            'texture': window.tolist(),
            'peak_time': float(time_axis[peak_idx]),
            'energy': float(energy[peak_idx]),
            'duration': float(len(window)/sr),
            'centroid_evolution': [float(c) for c in centroids] if centroids else [2000.0],
            'spectral_kurtosis': [float(k) for k in kurtoses] if kurtoses else [3.0]
        })
    
    all_centroids = [c for t in textures for c in t['centroid_evolution']]
    all_kurtoses = [k for t in textures for k in t['spectral_kurtosis']]
    
    spectral_dynamics = {
        'centroid_range': [float(np.percentile(all_centroids, 10)), 
                          float(np.percentile(all_centroids, 90))] if all_centroids else [1000.0, 4000.0],
        'kurtosis_stats': [float(x) for x in np.percentile(all_kurtoses, [25, 50, 75])] if all_kurtoses else [2.0, 3.0, 4.0]
    }
    
    print(f"🎵 Центроид: {spectral_dynamics['centroid_range'][0]:.0f}-{spectral_dynamics['centroid_range'][1]:.0f} Гц")
    
    nperseg = 4096
    f, Pxx = signal.welch(audio, fs=sr, nperseg=nperseg, scaling='density')
    Pxx = np.nan_to_num(Pxx, nan=0.0)
    
    window_length = min(51, len(Pxx) // 4)
    if window_length % 2 == 0:
        window_length -= 1
    Pxx_sg = savgol_filter(Pxx, window_length=max(3, window_length), polyorder=3) if window_length >= 3 else Pxx
    Pxx_med = signal.medfilt(Pxx, kernel_size=11)
    Pxx_gauss = ndimage.gaussian_filter1d(Pxx, sigma=2)
    
    Pxx_smooth = (Pxx_sg + Pxx_med + Pxx_gauss) / 3
    Pxx_smooth = np.maximum(Pxx_smooth, 1e-10)
    Pxx_norm = Pxx_smooth / (np.max(Pxx_smooth) + 1e-10)
    amp_envelope = np.sqrt(Pxx_norm)
    
    centroid = np.sum(f * Pxx) / (np.sum(Pxx) + 1e-10)
    
    cutoff_indices = np.where(Pxx_norm < 0.05)[0]
    main_peak_idx = np.argmax(Pxx_norm)
    cutoff_after = cutoff_indices[cutoff_indices > main_peak_idx]
    high_freq_cutoff = f[cutoff_after[0]] if len(cutoff_after) > 0 else (f[cutoff_indices[0]] if len(cutoff_indices) > 0 else 6500)
    high_freq_cutoff = max(high_freq_cutoff, 1000)
    
    peak_amplitude = np.max(np.abs(audio))
    rms_amplitude = np.sqrt(np.mean(audio**2))
    peak_rms_ratio = peak_amplitude / (rms_amplitude + 1e-10)
    
    print(f"📈 Peak/RMS: {peak_rms_ratio:.2f}, отсечка: {high_freq_cutoff:.0f} Гц")
    
    result = {
        'success': True,
        'audio': audio,
        'sr': sr,
        'duration': duration,
        'step_frequency': step_frequency,
        'contact_duration': contact_duration,
        'peak_rms_ratio': peak_rms_ratio,
        'high_freq_cutoff': high_freq_cutoff,
        'spectral_centroid': centroid,
        'f_spectrum': f,
        'Pxx_spectrum': Pxx_norm,
        'Pxx_methods': {'sg': Pxx_sg.tolist(), 'med': Pxx_med.tolist(), 'gauss': Pxx_gauss.tolist()},
        'amp_envelope': amp_envelope,
        'textures': textures,
        'spectral_dynamics': spectral_dynamics,
        'num_textures': len(textures)
    }
    
    save_cached_params(result)
    return result


# ========================
# ГЕНЕРАТОР V16 (АНТИ-АРТЕФАКТ)
# ========================

class RealisticSnowGenerator:
    def __init__(self, ref_params, sample_rate=44100):
        self.sr = sample_rate
        self.contact_duration = ref_params.get('contact_duration', 0.3)
        self.target_peak_rms = ref_params.get('peak_rms_ratio', 28.0)
        self.high_freq_cutoff = ref_params.get('high_freq_cutoff', 6500)
        self.spectral_dynamics = ref_params.get('spectral_dynamics', {'centroid_range': [1000, 4000]})
        
        self.crystal_coefficient = 0.45
        self.drag_factor = 1.0
        
        # 🔧 V16: Анти-артефакт параметры
        self.artifact_suppression = True
        self.dither_strength = 0.02
        
        self.ref_f = ref_params.get('f_spectrum')
        if ref_params.get('amp_envelope') is not None:
            self.amp_env = np.nan_to_num(ref_params.get('amp_envelope'), nan=0.0)
            self.spectral_interp = interpolate.interp1d(self.ref_f, self.amp_env, kind='linear', 
                                                        fill_value=0.0, bounds_error=False)
        else:
            self.spectral_interp = None
        
        textures_raw = ref_params.get('textures', [])
        self.textures = []
        for t in textures_raw:
            if isinstance(t, dict) and 'texture' in t:
                self.textures.append({
                    'texture': np.array(t['texture']) if isinstance(t['texture'], list) else t['texture'],
                    'peak_time': t.get('peak_time', 0),
                    'energy': t.get('energy', 1),
                    'duration': t.get('duration', 0.3),
                    'centroid_evolution': t.get('centroid_evolution', [2000]),
                    'spectral_kurtosis': t.get('spectral_kurtosis', [3])
                })
        
        if len(self.textures) > 0:
            print(f"✅ Загружено {len(self.textures)} текстур")
        
        self.phases = [0.05, 0.15, 0.10, 0.05]
        self.phase_names = ['impact', 'compression', 'shear', 'release']
        self.crossfade_samples = int(0.002 * self.sr)
        
    def _generate_spectral_shaped_noise(self, n_samples, target_centroid=None):
        n = max(10, int(n_samples))
        
        # 🔧 V16: Добавляем дитеринг для разбивания паттернов
        white = np.random.normal(0, 1, n) + np.random.uniform(-0.1, 0.1, n) * self.dither_strength
        
        X = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1/self.sr)
        
        if self.spectral_interp is not None:
            target_amp = self.spectral_interp(freqs)
            target_amp = np.nan_to_num(target_amp, nan=0.0)
            
            # 🔧 V16: Ещё более плавный rolloff (было 1200 → стало 1500)
            rolloff = np.ones_like(freqs)
            rolloff_start = self.high_freq_cutoff * 0.7
            rolloff_mask = freqs > rolloff_start
            if np.any(rolloff_mask):
                rolloff[rolloff_mask] = np.exp(-((freqs[rolloff_mask] - rolloff_start) / 1500)**2)
            
            target_amp = target_amp * rolloff
            
            hf_mask = freqs > self.high_freq_cutoff
            if np.any(hf_mask):
                target_amp[hf_mask] = np.maximum(target_amp[hf_mask], 0.18 * np.max(target_amp))
            
            if target_centroid is not None:
                centroid_boost = np.exp(-((freqs - target_centroid) / (target_centroid * 0.5))**2)
                target_amp = target_amp * (1 + 0.5 * centroid_boost)
            
            # 🔧 V16: Добавляем случайные вариации в амплитуду (разбивает гармоники)
            random_variation = np.random.uniform(0.85, 1.15, len(target_amp))
            target_amp = target_amp * random_variation
            
            X_shaped = X * target_amp
        else:
            X_shaped = X
        
        colored = np.fft.irfft(X_shaped, n=n)
        return np.nan_to_num(colored, nan=0.0)
    
    def _generate_squeak(self, n_samples, start_freq=800, end_freq=200, amplitude=0.3):
        n = max(10, int(n_samples))
        t = np.linspace(0, n / self.sr, n)
        freq = np.linspace(start_freq, end_freq, n)
        phase = 2 * np.pi * np.cumsum(freq) / self.sr
        
        # 🔧 V16: Добавляем частотную модуляцию для разбивания чистых гармоник
        fm_index = np.random.uniform(0.5, 2.0)
        fm_freq = np.random.uniform(50, 200)
        fm = np.sin(2 * np.pi * fm_freq * t) * fm_index
        
        squeak = amplitude * np.sin(phase + fm)
        squeak += amplitude * 0.5 * np.sin(2 * phase + 0.1 + fm * 0.5)
        squeak += amplitude * 0.25 * np.sin(3 * phase - 0.2 + fm * 0.25)
        envelope = np.hanning(n)
        result = squeak * envelope
        
        result = np.asarray(result).flatten()
        
        if len(result) != n:
            if len(result) > n:
                result = result[:n]
            else:
                result = np.pad(result, (0, n - len(result)), mode='constant')
        
        return result
    
    def _generate_shear_phase(self, n_samples, context_crispness=1.0, context_pressure=0.8):
        """Протяжная фаза скольжения с АНТИ-АРТЕФАКТ обработкой"""
        n = max(10, int(n_samples))
        
        noise = self._generate_spectral_shaped_noise(n, target_centroid=1500)
        noise = np.asarray(noise).flatten()[:n]
        if len(noise) < n:
            noise = np.pad(noise, (0, n - len(noise)), mode='constant')
        
        # 🔧 V16: Случайная модуляция вместо периодической (разбивает паттерны)
        mod_noise = np.random.normal(0, 1, n)
        b, a = butter(2, 800 / (self.sr/2), btype='low')
        mod_low = filtfilt(b, a, mod_noise)
        mod_low = 0.3 * mod_low / (np.std(mod_low) + 1e-10)
        
        # 🔧 V16: Уменьшенная глубина модуляции (было 0.15 → стало 0.1)
        mod = 1 + 0.1 * mod_low
        
        # 🔧 V16: Ещё меньше скрипа при экстремальных параметрах
        centroid_range = self.spectral_dynamics.get('centroid_range', [1000, 4000])
        start_freq = np.clip(centroid_range[1] * 0.5, 400, 800)   # Было 0.6 → стало 0.5
        end_freq = np.clip(centroid_range[0] * 0.3, 80, 250)      # Было 0.4 → стало 0.3
        
        # 🔧 V16: Ограничение скрипа при высоких значениях drag/pressure
        if self.drag_factor > 1.5 or context_pressure > 1.2:
            squeak_intensity = np.clip(context_pressure * 0.15, 0.08, 0.18)
            logger.info(f"🔧 Анти-артефакт: уменьшен скрип до {squeak_intensity:.3f}")
        else:
            squeak_intensity = np.clip(context_pressure * 0.2, 0.1, 0.25)
        
        squeak = self._generate_squeak(n, start_freq=start_freq, end_freq=end_freq, amplitude=squeak_intensity)
        squeak = np.asarray(squeak).flatten()[:n]
        if len(squeak) < n:
            squeak = np.pad(squeak, (0, n - len(squeak)), mode='constant')
        
        # 🔧 V16: Шум ещё больше доминирует (было 0.3-0.55 → стало 0.25-0.45)
        mix_curve = 0.25 + 0.2 * np.sin(np.linspace(0, np.pi, n))
        
        phase_noise = (1 - mix_curve) * noise * mod + mix_curve * squeak
        
        envelope = np.hanning(n)
        result = phase_noise * envelope
        result = np.asarray(result).flatten()
        
        if len(result) != n:
            if len(result) > n:
                result = result[:n]
            else:
                result = np.pad(result, (0, n - len(result)), mode='constant')
        
        return result
    
    def _phase_specific_noise(self, n_samples, phase_idx, context_crispness=1.0, context_pressure=0.8):
        n = max(10, int(n_samples))
        
        if phase_idx == 0:
            noise = self._generate_spectral_shaped_noise(n, target_centroid=3000)
            envelope = np.exp(-np.linspace(0, 4, n))
            return noise * envelope * np.hanning(n)[:n]
        
        elif phase_idx == 1:
            base = self._generate_spectral_shaped_noise(n, target_centroid=1500)
            crackle = np.ones(n)
            n_crackles = np.random.randint(5, 15)
            for _ in range(n_crackles):
                pos = np.random.randint(0, max(1, n-100))
                dur = np.random.randint(50, 300)
                end_pos = min(pos+dur, n)
                if end_pos > pos:
                    crackle[pos:end_pos] *= np.random.uniform(1.2, 2.0)
            return base * crackle * context_crispness
        
        elif phase_idx == 2:
            return self._generate_shear_phase(n, context_crispness, context_pressure)
        
        else:
            noise = np.random.normal(0, 0.1, n)
            return noise * np.exp(-np.linspace(0, 6, n))
    
    def _generate_crystal_impulses(self, n_crystals, n_samples):
        impulses = np.zeros(n_samples)
        used_positions = []
        
        for _ in range(n_crystals):
            attempts = 0
            while attempts < 50:
                pos = np.random.randint(0, max(1, n_samples-20))
                dur = np.random.randint(5, 15)
                
                overlap = False
                for used_pos, used_dur in used_positions:
                    if abs(pos - used_pos) < (dur + used_dur) // 2:
                        overlap = True
                        break
                
                if not overlap:
                    break
                attempts += 1
            
            if attempts < 50:
                amp = np.random.uniform(0.1, 0.4)
                impulse = np.random.normal(0, 1, dur) * np.hanning(dur) * amp
                
                end_pos = min(pos + dur, n_samples)
                actual_dur = end_pos - pos
                if actual_dur > 0:
                    impulses[pos:end_pos] += impulse[:actual_dur]
                    used_positions.append((pos, dur))
        
        return impulses
    
    def _crossfade(self, signal1, signal2, fade_samples):
        signal1 = np.asarray(signal1).flatten()
        signal2 = np.asarray(signal2).flatten()
        
        fade_samples = int(fade_samples)
        if fade_samples <= 0:
            return signal2
        
        if len(signal1) < fade_samples or len(signal2) < fade_samples:
            fade_samples = min(len(signal1), len(signal2), fade_samples)
            if fade_samples <= 0:
                return signal2
        
        fade_in = np.linspace(0, 1, fade_samples) ** 2
        fade_out = np.linspace(1, 0, fade_samples) ** 2
        
        result = signal2.copy()
        result[:fade_samples] = signal1[:fade_samples] * fade_out + signal2[:fade_samples] * fade_in
        
        return result
    
    def _soft_limiter(self, signal, threshold=0.95):
        abs_signal = np.abs(signal)
        mask = abs_signal > threshold
        
        if np.any(mask):
            signal[mask] = np.sign(signal[mask]) * (threshold + (abs_signal[mask] - threshold) / (1 + abs_signal[mask] - threshold))
        
        return signal
    
    def _add_decay_tail(self, signal, tail_duration=0.15):
        signal = np.asarray(signal)
        tail_samples = int(tail_duration * self.sr)
        if tail_samples <= 0:
            return signal
        
        last_value = float(signal[-1]) if len(signal) > 0 else 0.0
        tail = last_value * np.exp(-np.linspace(0, 5, tail_samples))
        
        return np.concatenate([signal, tail])
    
    def _apply_spectral_smoothing(self, signal, n_samples):
        """🔧 V16: Сглаживание спектра для устранения артефактов"""
        if not self.artifact_suppression:
            return signal
        
        # FFT
        X = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n_samples, 1/self.sr)
        
        # Сглаживание амплитуды (скользящее среднее)
        magnitude = np.abs(X)
        phase = np.angle(X)
        
        # Скользящее среднее для разбивания резких пиков
        window_size = max(3, len(magnitude) // 200)
        if window_size % 2 == 0:
            window_size += 1
        
        magnitude_smooth = np.convolve(magnitude, np.ones(window_size)/window_size, mode='same')
        
        # Смешивание оригинала и сглаженного (70/30)
        magnitude_final = 0.7 * magnitude + 0.3 * magnitude_smooth
        
        # Обратное FFT
        X_smooth = magnitude_final * np.exp(1j * phase)
        signal_smooth = np.fft.irfft(X_smooth, n=n_samples)
        
        return signal_smooth
    
    def generate_crunch_event(self, context_crispness=1.0, context_depth=0.15, context_pressure=0.8):
        try:
            # 🔧 V16: Проверка на экстремальные параметры
            is_extreme = (self.drag_factor > 1.5 or context_depth > 0.35 or 
                         context_pressure > 1.2 or context_crispness > 1.5)
            
            if is_extreme:
                logger.info(f"🔧 Экстремальные параметры detected! Включаю анти-артефакт режим")
                old_drag = self.drag_factor
                self.drag_factor = np.clip(self.drag_factor, 0.5, 1.6)  # Ограничение
                logger.info(f"   Тягучесть ограничена: {old_drag:.2f} → {self.drag_factor:.2f}")
            
            if len(self.textures) > 0:
                texture_idx = np.random.randint(0, len(self.textures))
                base_texture = np.asarray(self.textures[texture_idx]['texture']).flatten()
            else:
                base_texture = None
                
            depth_factor = np.clip(context_depth / 0.15, 0.5, 2.0)
            pressure_factor = np.clip(context_pressure / 0.8, 0.8, 1.5)
            
            adjusted_phases = [
                self.phases[0] * (1 + 0.1 * (depth_factor - 1)),
                self.phases[1] * (1 + 0.2 * (pressure_factor - 1)),
                self.phases[2] * (1 + 0.5 * (pressure_factor - 1) * depth_factor * self.drag_factor),
                self.phases[3] * (1 + 0.3 * (depth_factor - 1))
            ]
            total_phase_time = sum(adjusted_phases)
            
            n_samples = int(total_phase_time * self.sr)
            crunch = np.zeros(n_samples)
            
            phase_samples = [int(p * self.sr) for p in adjusted_phases]
            
            current_sample = 0
            prev_phase_end = None
            
            for i, (phase_name, length) in enumerate(zip(self.phase_names, phase_samples)):
                remaining_samples = n_samples - current_sample
                if remaining_samples <= 0:
                    break
                
                actual_length = min(length, remaining_samples)
                
                if i == 2:
                    phase_noise = self._generate_shear_phase(actual_length, context_crispness, context_pressure)
                else:
                    phase_noise = self._phase_specific_noise(actual_length, i, context_crispness, context_pressure)
                
                phase_noise = np.asarray(phase_noise).flatten()
                
                if len(phase_noise) > actual_length:
                    phase_noise = phase_noise[:actual_length]
                elif len(phase_noise) < actual_length:
                    phase_noise = np.pad(phase_noise, (0, actual_length - len(phase_noise)), mode='constant')
                
                if base_texture is not None and current_sample < len(base_texture):
                    texture_end = min(current_sample + actual_length, len(base_texture))
                    texture_segment = base_texture[current_sample:texture_end]
                    
                    if len(texture_segment) == len(phase_noise):
                        blend = 0.5 + 0.3 * np.random.random()
                        phase_noise = blend * phase_noise + (1 - blend) * texture_segment
                    elif len(texture_segment) < len(phase_noise):
                        texture_segment = np.pad(texture_segment, (0, len(phase_noise) - len(texture_segment)), mode='constant')
                        blend = 0.5 + 0.3 * np.random.random()
                        phase_noise = blend * phase_noise + (1 - blend) * texture_segment
                
                if prev_phase_end is not None and self.crossfade_samples > 0:
                    prev_phase_end = np.asarray(prev_phase_end).flatten()
                    fade_len = min(self.crossfade_samples, len(phase_noise) // 4, len(prev_phase_end))
                    if fade_len > 0:
                        phase_noise = self._crossfade(prev_phase_end[:fade_len], phase_noise, fade_len)
                        if len(phase_noise) > actual_length:
                            phase_noise = phase_noise[:actual_length]
                
                copy_len = min(len(phase_noise), remaining_samples)
                crunch[current_sample:current_sample + copy_len] = phase_noise[:copy_len]
                
                prev_phase_end = phase_noise[-min(self.crossfade_samples, len(phase_noise)):]
                current_sample += copy_len
            
            crystal_impulses = self._generate_crystal_impulses(n_samples // 10, n_samples)
            crystal_impulses = np.asarray(crystal_impulses).flatten()
            
            if len(crystal_impulses) == len(crunch):
                crunch += self.crystal_coefficient * crystal_impulses
            elif len(crystal_impulses) > len(crunch):
                crunch += self.crystal_coefficient * crystal_impulses[:len(crunch)]
            else:
                crunch[:len(crystal_impulses)] += self.crystal_coefficient * crystal_impulses
            
            if context_crispness > 1.2:
                crunch *= 1.15
            elif context_crispness < 0.8:
                b, a = butter(2, 0.3, btype='low')
                crunch = filtfilt(b, a, crunch) * 0.8
            
            crunch = self._soft_limiter(crunch, threshold=0.85)
            
            crunch_before_tail = len(crunch)
            crunch = self._add_decay_tail(crunch, tail_duration=0.15)
            
            n_samples = len(crunch)
            
            # 🔧 V16: Применяем спектральное сглаживание
            crunch = self._apply_spectral_smoothing(crunch, n_samples)
            
            current_peak_rms = np.max(np.abs(crunch)) / (np.sqrt(np.mean(crunch ** 2)) + 1e-10)
            
            if current_peak_rms > 0 and self.target_peak_rms > 0:
                target_ratio = self.target_peak_rms
                if current_peak_rms < target_ratio * 0.7:
                    current_len = len(crunch)
                    extra = int(current_len * 0.3)
                    
                    extended = np.zeros(current_len + extra)
                    extended[:current_len] = crunch
                    extended[current_len:] = crunch[-1] * np.exp(-np.linspace(0, 4, extra))
                    crunch = extended
            
            max_val = np.max(np.abs(crunch))
            if max_val > 1e-10:
                crunch = crunch / max_val * 0.9
            
            if is_extreme:
                self.drag_factor = old_drag
            
            return crunch
            
        except Exception as e:
            logger.error(f"❌ Ошибка в generate_crunch_event: {e}", exc_info=True)
            raise
    
    def set_drag_factor(self, drag):
        self.drag_factor = np.clip(drag, 0.5, 2.0)


# ========================
# ФИЗИЧЕСКИЙ КОНТЕКСТ
# ========================

class SnowPhysicsContext:
    def __init__(self):
        self.state = {
            'snow_depth': 0.15,
            'temperature': -5.0,
            'walk_speed': 1.2,
            'foot_pressure': 0.8,
            'drag_factor': 1.0
        }
    
    def update_context(self, step_num):
        self.state['snow_depth'] *= 0.997
        crispness = max(0.5, min(2.0, (self.state['temperature'] + 15) / 10))
        return {
            'crispness': crispness, 
            'depth': self.state['snow_depth'],
            'pressure': self.state['foot_pressure'],
            'drag': self.state['drag_factor']
        }


# ========================
# СИМУЛЯТОР V16
# ========================

class SnowWalkSimulatorV16:
    def __init__(self, ref_params, sim_duration=20, sample_rate=44100):
        self.sr = sample_rate
        self.sim_duration = sim_duration
        self.step_frequency = ref_params.get('step_frequency', 1.8)
        self.generator = RealisticSnowGenerator(ref_params, sample_rate)
        self.physics = SnowPhysicsContext()
        
        interval = 1.0 / self.step_frequency
        self.contact_duration = ref_params.get('contact_duration', 0.3)
        self.step_duration = self.contact_duration * 0.75
        self.pause_duration = interval * 0.6
        
        duty = self.step_duration / interval * 100 if interval > 0 else 0
        print(f"   Шаг: {self.step_duration:.3f} с, пауза: {self.pause_duration:.3f} с")
        print(f"   📊 Duty cycle: {duty:.0f}% звук, {100-duty:.0f}% тишина")
    
    def simulate(self):
        total_samples = int(self.sim_duration * self.sr)
        audio = np.zeros(total_samples, dtype=np.float64)
        
        base_interval = 1.0 / self.step_frequency
        current_time = 0.0
        step_times = []
        step_count = 0
        
        while current_time < self.sim_duration:
            interval = base_interval * np.random.uniform(0.92, 1.08)
            context = self.physics.update_context(step_count)
            
            self.generator.set_drag_factor(context['drag'])
            
            try:
                crunch = self.generator.generate_crunch_event(
                    context['crispness'], 
                    context['depth'],
                    context['pressure']
                )
            except Exception as e:
                logger.error(f"❌ Ошибка генерации шага #{step_count + 1}: {e}")
                raise
            
            start = int(current_time * self.sr)
            end = min(start + len(crunch), total_samples)
            
            if start < total_samples:
                audio[start:end] += crunch[:end-start] * np.random.uniform(0.85, 1.15)
            
            step_times.append(current_time)
            step_count += 1
            current_time += interval
        
        audio = np.nan_to_num(audio, nan=0.0)
        if np.max(np.abs(audio)) > 1e-10:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        print(f"✅ Сгенерировано {step_count} шагов")
        return audio, step_times


# ========================
# GUI V16
# ========================

class InteractiveSnowGUI:
    def __init__(self, ref_params):
        self.ref_params = ref_params
        self.root = tk.Tk()
        self.root.title("Снежный хруст V16 – Анти-артефакт")
        self.root.geometry("900x750")
        
        self.temperature = tk.DoubleVar(value=-5.0)
        self.snow_depth = tk.DoubleVar(value=0.15)
        self.walk_speed = tk.DoubleVar(value=1.8)
        self.pressure = tk.DoubleVar(value=0.8)
        self.drag_factor = tk.DoubleVar(value=1.0)
        self.crystal_intensity = tk.DoubleVar(value=0.45)
        self.record_duration = tk.IntVar(value=20)
        
        self.context = None
        self.simulator = None
        self.is_playing = False
        self.current_audio = None
        self.current_sr = ref_params.get('sr', 44100)
        
        self.create_widgets()
        self.update_parameters()
    
    def create_widgets(self):
        params_frame = ttk.LabelFrame(self.root, text="Параметры снега", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        row = 0
        ttk.Label(params_frame, text="Температура (°C):").grid(row=row, column=0, sticky=tk.W, pady=3)
        scale = ttk.Scale(params_frame, from_=-25, to=0, variable=self.temperature, command=self.on_change)
        scale.grid(row=row, column=1, padx=10, pady=3, sticky=(tk.W, tk.E))
        ttk.Label(params_frame, textvariable=self.temperature, width=6).grid(row=row, column=2)
        row += 1
        
        ttk.Label(params_frame, text="Глубина снега (м):").grid(row=row, column=0, sticky=tk.W, pady=3)
        scale = ttk.Scale(params_frame, from_=0.05, to=0.5, variable=self.snow_depth, command=self.on_change)
        scale.grid(row=row, column=1, padx=10, pady=3, sticky=(tk.W, tk.E))
        ttk.Label(params_frame, textvariable=self.snow_depth, width=6).grid(row=row, column=2)
        row += 1
        
        ttk.Label(params_frame, text="Частота шагов (Гц):").grid(row=row, column=0, sticky=tk.W, pady=3)
        scale = ttk.Scale(params_frame, from_=0.8, to=2.5, variable=self.walk_speed, command=self.on_change)
        scale.grid(row=row, column=1, padx=10, pady=3, sticky=(tk.W, tk.E))
        ttk.Label(params_frame, textvariable=self.walk_speed, width=6).grid(row=row, column=2)
        row += 1
        
        ttk.Label(params_frame, text="Давление ноги:").grid(row=row, column=0, sticky=tk.W, pady=3)
        scale = ttk.Scale(params_frame, from_=0.2, to=1.5, variable=self.pressure, command=self.on_change)
        scale.grid(row=row, column=1, padx=10, pady=3, sticky=(tk.W, tk.E))
        ttk.Label(params_frame, textvariable=self.pressure, width=6).grid(row=row, column=2)
        row += 1
        
        ttk.Label(params_frame, text="🌀 Тягучесть:").grid(row=row, column=0, sticky=tk.W, pady=3)
        scale = ttk.Scale(params_frame, from_=0.5, to=2.0, variable=self.drag_factor, command=self.on_change)
        scale.grid(row=row, column=1, padx=10, pady=3, sticky=(tk.W, tk.E))
        ttk.Label(params_frame, textvariable=self.drag_factor, width=6).grid(row=row, column=2)
        row += 1
        
        ttk.Label(params_frame, text="💎 Хруст кристаллов:").grid(row=row, column=0, sticky=tk.W, pady=3)
        scale = ttk.Scale(params_frame, from_=0.2, to=0.8, variable=self.crystal_intensity, command=self.on_change)
        scale.grid(row=row, column=1, padx=10, pady=3, sticky=(tk.W, tk.E))
        ttk.Label(params_frame, textvariable=self.crystal_intensity, width=6).grid(row=row, column=2)
        
        viz_frame = ttk.LabelFrame(self.root, text="Визуализация спектра", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig = Figure(figsize=(8, 4), dpi=100)
        
        self.ax_waveform = self.fig.add_subplot(211)
        self.ax_waveform.set_title('Амплитудный спектр (осциллограмма)', fontsize=10)
        self.ax_waveform.set_xlabel('Время (с)')
        self.ax_waveform.set_ylabel('Амплитуда')
        self.ax_waveform.grid(True, alpha=0.3)
        self.waveform_line, = self.ax_waveform.plot([], [], linewidth=0.5, color='#2196F3')
        
        self.ax_spectrum = self.fig.add_subplot(212)
        self.ax_spectrum.set_title('Частотный спектр (FFT) — V16 без артефактов', fontsize=10)
        self.ax_spectrum.set_xlabel('Частота (Гц)')
        self.ax_spectrum.set_ylabel('Мощность (дБ)')
        self.ax_spectrum.grid(True, alpha=0.3)
        self.spectrum_line, = self.ax_spectrum.plot([], [], linewidth=1, color='#FF5722')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        dur_frame = ttk.Frame(control_frame)
        dur_frame.pack(side=tk.LEFT)
        ttk.Label(dur_frame, text="📼 Длительность (с):").pack(side=tk.LEFT)
        dur_spinbox = ttk.Spinbox(dur_frame, from_=5, to=120, width=5, textvariable=self.record_duration)
        dur_spinbox.pack(side=tk.LEFT, padx=10)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.LEFT, padx=20)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ Play", command=self.play)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self.stop)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(btn_frame, text="💾 Сохранить", command=self.save_audio)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.update_spectrum_btn = ttk.Button(btn_frame, text="🔄 Обновить спектр", command=self.update_spectrum_display)
        self.update_spectrum_btn.pack(side=tk.LEFT, padx=5)
        
        self.status = ttk.Label(self.root, text="Готов", relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.status.pack(fill=tk.X, padx=10, pady=5)
        
        if not SOUNDDEVICE_AVAILABLE:
            ttk.Label(self.root, text="⚠️ sounddevice не установлен – только сохранение", 
                     foreground='orange').pack(pady=2)
        
        ttk.Label(self.root, text="💡 V16: Спектральное сглаживание + дитеринг устраняют артефакты", 
                 font=('TkDefaultFont', 8), foreground='gray').pack(pady=2)
    
    def update_spectrum_display(self, audio=None):
        if audio is None:
            audio = self.current_audio
        
        if audio is None or len(audio) == 0:
            logger.info("Нет аудио для отображения спектра")
            return
        
        try:
            display_samples = min(int(0.1 * self.current_sr), len(audio))
            time_axis = np.arange(display_samples) / self.current_sr
            self.waveform_line.set_data(time_axis, audio[:display_samples])
            self.ax_waveform.set_xlim(0, time_axis[-1])
            amplitude_max = np.max(np.abs(audio[:display_samples]))
            self.ax_waveform.set_ylim(-amplitude_max * 1.1, amplitude_max * 1.1)
            
            n_fft = min(4096, len(audio))
            fft_data = np.fft.rfft(audio[:n_fft])
            freqs = np.fft.rfftfreq(n_fft, 1/self.current_sr)
            magnitude = 20 * np.log10(np.abs(fft_data) + 1e-10)
            
            self.spectrum_line.set_data(freqs, magnitude)
            self.ax_spectrum.set_xlim(0, min(10000, self.current_sr/2))
            self.ax_spectrum.set_ylim(np.min(magnitude), np.max(magnitude) + 10)
            
            self.canvas.draw_idle()
            
            logger.info(f"✅ Спектр обновлён: {len(audio)} сэмплов, {self.current_sr} Гц")
            
        except Exception as e:
            logger.error(f"❌ Ошибка обновления спектра: {e}")
    
    def on_change(self, event=None):
        logger.info(f"🎚️  Изменение параметра через GUI")
        self.update_parameters()
    
    def update_parameters(self):
        logger.info("⚙️  Обновление параметров симулятора")
        
        if self.context is None:
            self.context = SnowPhysicsContext()
        
        self.context.state['temperature'] = self.temperature.get()
        self.context.state['snow_depth'] = self.snow_depth.get()
        self.context.state['walk_speed'] = self.walk_speed.get()
        self.context.state['foot_pressure'] = self.pressure.get()
        self.context.state['drag_factor'] = self.drag_factor.get()
        
        self.simulator = SnowWalkSimulatorV16(
            {**self.ref_params, 'step_frequency': self.walk_speed.get()},
            sim_duration=5,
            sample_rate=self.ref_params.get('sr', 44100)
        )
        self.simulator.physics = self.context
        self.simulator.generator.crystal_coefficient = self.crystal_intensity.get()
        
        self.status.config(
            text=f"T={self.temperature.get():.1f}°C | глубина={self.snow_depth.get():.2f}м | "
                 f"скорость={self.walk_speed.get():.2f} Гц | тягучесть={self.drag_factor.get():.2f} | "
                 f"хруст={self.crystal_intensity.get():.2f}"
        )
    
    def generate_audio(self, duration=None):
        if self.simulator:
            if duration is not None:
                old_duration = self.simulator.sim_duration
                self.simulator.sim_duration = duration
                audio, _ = self.simulator.simulate()
                self.simulator.sim_duration = old_duration
                return audio
            else:
                audio, _ = self.simulator.simulate()
                return audio
        return None
    
    def play(self):
        if self.is_playing:
            return
        
        def _play_thread():
            self.is_playing = True
            self.status.config(text="⏳ Генерация...")
            self.root.update_idletasks()
            
            try:
                audio = self.generate_audio(duration=5)
                if audio is not None and len(audio) > 0:
                    self.current_audio = audio
                    self.status.config(text="🔊 Воспроизведение...")
                    self.root.update_idletasks()
                    
                    self.update_spectrum_display(audio)
                    
                    if SOUNDDEVICE_AVAILABLE:
                        sd.play(audio, samplerate=self.current_sr)
                        sd.wait()
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        wavfile.write(f"snow_preview_{timestamp}.wav", 
                                     self.current_sr,
                                     (audio * 32767).astype(np.int16))
                        self.status.config(text=f"💾 Сохранено: snow_preview_{timestamp}.wav")
                    
                    self.status.config(text="✅ Готов")
                else:
                    self.status.config(text="❌ Ошибка генерации")
            except Exception as e:
                self.status.config(text=f"❌ Ошибка: {str(e)[:30]}")
            finally:
                self.is_playing = False
        
        threading.Thread(target=_play_thread, daemon=True).start()
    
    def stop(self):
        if SOUNDDEVICE_AVAILABLE:
            sd.stop()
        self.is_playing = False
        self.status.config(text="⏹ Остановлено")
    
    def save_audio(self):
        duration = self.record_duration.get()
        
        def _save_thread():
            self.status.config(text=f"⏳ Генерация ({duration} с)...")
            self.root.update_idletasks()
            
            try:
                audio = self.generate_audio(duration=duration)
                
                if audio is None or len(audio) == 0:
                    self.status.config(text="❌ Ошибка генерации")
                    return
                
                self.current_audio = audio
                self.update_spectrum_display(audio)
                
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".wav",
                    filetypes=[("WAV файлы", "*.wav"), ("Все файлы", "*.*")],
                    initialfile=f"snow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                    title="Сохранить аудио как..."
                )
                
                if not file_path:
                    self.status.config(text="↪ Отменено")
                    return
                
                sr = self.ref_params.get('sr', 44100)
                audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                wavfile.write(file_path, sr, audio_int16)
                
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                actual_duration = len(audio) / sr
                
                self.status.config(
                    text=f"✅ Сохранено: {os.path.basename(file_path)} "
                         f"({actual_duration:.1f} с, {file_size_mb:.1f} MB)"
                )
                
            except Exception as e:
                self.status.config(text=f"❌ Ошибка: {str(e)[:40]}")
        
        threading.Thread(target=_save_thread, daemon=True).start()
    
    def run(self):
        self.root.mainloop()


# ========================
# ЗАПУСК
# ========================

def main():
    print("=" * 60)
    print("СИМУЛЯЦИЯ ХРУСТА СНЕГА V16 (АНТИ-АРТЕФАКТ)")
    print("=" * 60)
    
    ref_params = load_cached_params('ref_params_v16.json')
    
    if ref_params is None or not ref_params.get('success', False):
        ref_file = "C:\\s.mp3"
        for path in ["C:\\s.mp3", "C:\\S.mp3", "s.mp3", "S.mp3"]:
            if os.path.exists(path):
                ref_file = path
                break
        
        print(f"\n📁 Файл: {ref_file}")
        print("\n📊 Анализ эталона...")
        ref_params = analyze_reference_enhanced(ref_file)
    
    if not ref_params.get('success', False):
        print("❌ Не удалось загрузить эталон.")
        return None, None, None
    
    print("\n🖥️  Запуск интерактивного GUI...")
    gui = InteractiveSnowGUI(ref_params)
    gui.run()
    
    return None, None, None


if __name__ == "__main__":
    main()