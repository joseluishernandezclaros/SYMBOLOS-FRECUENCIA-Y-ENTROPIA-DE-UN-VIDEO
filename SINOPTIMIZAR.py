import cv2
import numpy as np
import pandas as pd
import random
import string
import matplotlib.pyplot as plt


def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / float(hist.sum() + 1e-6)
    entropy = -np.sum(hist * np.log2(hist + 1e-6))
    return entropy


def generate_all_symbols(frame):
    symbols = []

    for row in frame:
        for pixel in row:
            symbol = tuple(pixel)
            symbols.append(symbol)

    return symbols


# Nombre del archivo de video
video_path = 'video_examples_comprimido\input.mkv'

# Abre el video
cap = cv2.VideoCapture(video_path)

# Inicializa diccionarios para almacenar la cuenta de cada símbolo y la entropía
symbol_counts_dict = {}
entropy_dict = {}

# Lee los fotogramas del video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    entropy = calculate_entropy(gray)
    print(f'Entropía para el frame {current_frame + 1}: {entropy}')

    entropy_dict[current_frame] = entropy

    all_symbols = generate_all_symbols(frame)

    for symbol in all_symbols:
        symbol_key = str(symbol)
        if symbol_key in symbol_counts_dict:
            symbol_counts_dict[symbol_key] += 1
        else:
            symbol_counts_dict[symbol_key] = 1

    current_frame += 1
    print(f'Frames procesados: {current_frame}/{total_frames}')

cap.release()

# Calcular la entropía para el conjunto completo de símbolos
all_symbol_counts = list(symbol_counts_dict.values())
total_symbols = sum(all_symbol_counts)
entropy_all_symbols = -sum(count/total_symbols * np.log2(count /
                           total_symbols + 1e-6) for count in all_symbol_counts)

# Calcular la entropía por cada símbolo basada en las frecuencias
symbol_entropies = {}
for symbol, count in symbol_counts_dict.items():
    symbol_prob = count / total_symbols
    symbol_entropies[symbol] = -symbol_prob * np.log2(symbol_prob + 1e-6)

# Calcular la probabilidad para cada símbolo
symbol_probabilities = {
    symbol: count / total_symbols for symbol, count in symbol_counts_dict.items()}

# Crear un DataFrame con las frecuencias de los símbolos y sus probabilidades
df_frequencies = pd.DataFrame({'Símbolo': list(symbol_counts_dict.keys()), 'Frecuencia': list(
    symbol_counts_dict.values()), 'Probabilidad': list(symbol_probabilities.values())})

# Crear un DataFrame con las entropías por símbolo
df_entropies = pd.DataFrame(symbol_entropies.items(), columns=[
                            'Símbolo', 'Entropía por Símbolo'])

# Mostrar los DataFrames en la consola
print("DataFrame de Frecuencias de Símbolos:")
print(df_frequencies)

print("\nDataFrame de Entropías por Símbolo:")
print(df_entropies)
# Guardar el DataFrame de Frecuencias en un archivo de texto
df_frequencies.to_csv('df_frequencies.txt', index=False, sep='\t')

# Guardar el DataFrame de Entropías por Símbolo en un archivo de texto
# Unir los DataFrames df_frequencies y df_entropies
df_entropies.to_csv('df_entropies.txt', index=False, sep='\t')
df_combined = pd.concat([df_frequencies, df_entropies], axis=1)
df_entropies.drop(columns=['Símbolo'], inplace=True)
# Mostrar el DataFrame combinado en la consola
print("DataFrame Combinado:")
print(df_combined)

# Guardar el DataFrame combinado en un archivo de texto
df_combined.to_csv('df_combined.txt', index=False, sep='\t')
