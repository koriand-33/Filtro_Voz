import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# Cargar archivo de audio
data, fs = sf.read('audioprueba.wav')  

if len(data.shape) == 2:
    data = data.mean(axis=1)


data = data / np.max(np.abs(data))

print(f"Duración: {len(data)/fs:.2f} segundos, Frecuencia de muestreo: {fs} Hz")

# Duración total del audio (en segundos)
duracion = len(data) / fs
print(f"Duración total: {duracion:.2f} segundos")

# Elegimos una ventana de tiempo para visualizar (por ejemplo, los primeros 2 segundos)
tiempo = np.linspace(0, len(data)/fs, len(data))  # vector de tiempo

plt.figure(figsize=(10, 4))
plt.plot(tiempo[:int(2*fs)], data[:int(2*fs)], label='Señal original con ruido')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal de audio (dominio del tiempo)')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

orden = 4
fc = 5000  # Frecuencia de corte en Hz
Wn = fc / (fs / 2)  # Frecuencia normalizada

# Crear el filtro pasa bajos Butterworth
b, a = butter(N=orden, Wn=Wn, btype='low')

# Aplicar el filtro con filtfilt (para no introducir desfase)
data_filtrada = filtfilt(b, a, data)

sf.write('voz_filtrada.wav', data_filtrada, fs)
print("✅ Audio filtrado guardado como voz_filtrada.wav")

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
