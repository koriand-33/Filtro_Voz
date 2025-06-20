import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz
from scipy.fft import fft, fftfreq
from pedalboard import Pedalboard, Reverb, Compressor
from pedalboard.io import AudioFile
from pedalboard import *


# === 1. Cargar archivo de audio ===
data, fs = sf.read('audio2.wav')
if len(data.shape) == 2:
    data = data.mean(axis=1)  # Convertir a mono si es estéreo

data = data / np.max(np.abs(data))  # Normalizar
print(f"Duración: {len(data)/fs:.2f} s, Frecuencia de muestreo: {fs} Hz")

# === 2. Visualizar la señal original ===
tiempo = np.linspace(0, len(data)/fs, len(data))
plt.figure(figsize=(10, 4))
plt.plot(tiempo[:int(2*fs)], data[:int(2*fs)], label='Señal original con ruido')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal de audio - Dominio del tiempo')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === 3. Crear y aplicar el filtro pasa bajos ===
orden = 5
fc = 5000  # Frecuencia de corte en Hz
Wn = fc / (fs / 2)  # Frecuencia normalizada
b, a = butter(N=orden, Wn=Wn, btype='low')
data_filtrada = filtfilt(b, a, data)

# === 4. Guardar archivo filtrado ===
sf.write('voz_filtrada2.wav', data_filtrada, fs)
print("✅ Audio filtrado guardado como voz_filtrada.wav")

# === 5. Comparar en tiempo ===
plt.figure(figsize=(10, 4))
plt.plot(tiempo[:int(2*fs)], data[:int(2*fs)], label='Original', alpha=0.5)
plt.plot(tiempo[:int(2*fs)], data_filtrada[:int(2*fs)], label='Filtrada', color='orange')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Comparación antes y después del filtro')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === 6. Visualizar espectro de frecuencia ===
N = len(data)
f = fftfreq(N, 1/fs)
fft_original = np.abs(fft(data)) / N
fft_filtrada = np.abs(fft(data_filtrada)) / N
f = f[:N//2]
fft_original = fft_original[:N//2]
fft_filtrada = fft_filtrada[:N//2]

plt.figure(figsize=(10, 4))
plt.plot(f, fft_original, label='Original', alpha=0.6)
plt.plot(f, fft_filtrada, label='Filtrada', color='orange')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.title('Espectro de frecuencia (FFT)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xlim(0, fs//2)
plt.show()

# === 7. Espectrograma ===
plt.figure(figsize=(10, 4))
plt.specgram(data, NFFT=1024, Fs=fs, noverlap=512, cmap='viridis')
plt.colorbar(label='Intensidad [dB]')
plt.title('Espectrograma - Señal original')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.specgram(data_filtrada, NFFT=1024, Fs=fs, noverlap=512, cmap='viridis')
plt.colorbar(label='Intensidad [dB]')
plt.title('Espectrograma - Señal filtrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.tight_layout()
plt.show()

# === 8. Respuesta en frecuencia del filtro ===
w, h = freqz(b, a, fs=fs)
plt.figure(figsize=(8, 3))
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Respuesta en frecuencia del filtro')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Ganancia [dB]')
plt.grid()
plt.tight_layout()
plt.show()

# === 9. Energía antes y después ===
energia_original = np.sum(data**2)
energia_filtrada = np.sum(data_filtrada**2)
print(f"Energía original: {energia_original:.2f}")
print(f"Energía filtrada: {energia_filtrada:.2f}")
print(f"Reducción de energía: {100 * (1 - energia_filtrada/energia_original):.2f}%")

# === 10. Guardar versión amplificada ===
max_amplitud = np.max(np.abs(data_filtrada))
if max_amplitud > 0:
    amplificado = 0.95 * data_filtrada / max_amplitud
    sf.write('voz_filtrada_amplificada2.wav', amplificado, fs)
    print("✅ Versión amplificada guardada como voz_filtrada_amplificada2.wav")



# Aplicar efectos con Pedalboard (compresor + reverb)


with AudioFile('voz_filtrada2.wav') as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate

board = Pedalboard([
    Compressor(threshold_db=15, ratio=1.5,release_ms=250),
    NoiseGate(threshold_db=20,ratio=1.5,release_ms=250),
    Gain(gain_db=10),
])

audio_efectos = board(audio, samplerate)

with AudioFile('voz_filtrada_efectos2.wav', 'w', samplerate, audio.shape[0]) as f:
    f.write(audio_efectos)

print("🎛️ Audio con efectos guardado como voz_filtrada_efectos.wav")


#Visualización de la señal con efectos
# Vector de tiempo para la señal con efectos
tiempo_efectos = np.linspace(0, len(audio_efectos)/fs, len(audio_efectos))

plt.figure(figsize=(10, 4))
plt.plot(tiempo[:int(2*fs)], data[:int(2*fs)], label='Original', alpha=0.3)
plt.plot(tiempo[:int(2*fs)], data_filtrada[:int(2*fs)], label='Filtrada', alpha=0.5)
plt.plot(tiempo_efectos[:int(2*fs)], audio_efectos[:int(2*fs)], label='Con efectos', color='green')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Comparación: Original vs Filtrada vs Efectos')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#Espectro de frecuencia 
fft_efectos = np.abs(fft(audio_efectos)) / len(audio_efectos)
fft_efectos = fft_efectos[:N//2]  # solo parte positiva

plt.figure(figsize=(10, 4))
plt.plot(f, fft_original, label='Original', alpha=0.4)
plt.plot(f, fft_filtrada, label='Filtrada', alpha=0.6)
plt.plot(f, fft_efectos, label='Con efectos', color='green')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud')
plt.title('Espectro de frecuencia después de efectos')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0, fs//2)
plt.show()


#Espectograma de la señal con efectos 
plt.figure(figsize=(10, 4))
plt.specgram(audio_efectos, NFFT=1024, Fs=fs, noverlap=512, cmap='plasma')
plt.colorbar(label='Intensidad [dB]')
plt.title('Espectrograma - Señal con efectos')
plt.xlabel('Tiempo [s]')
plt.ylabel('Frecuencia [Hz]')
plt.tight_layout()
plt.show()
