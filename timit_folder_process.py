import os
import soundfile as sf
from tqdm import tqdm

import os
import soundfile as sf
from collections import Counter, defaultdict
from tqdm import tqdm

# Constantes
TARGET_SAMPLING_RATE = 16000
N_FFT = 400
OUTPUT_DIR = "output_segments"
MIN_OCCURRENCES = 10
MAX_OCCURRENCES = 30

def extract_words_from_timit(dataset_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split_dirs = ['TRAIN', 'TEST']
    word_counts = Counter()
    wrd_files = []

    # 🔁 Première passe : compter les occurrences de tous les mots
    for split in split_dirs:
        split_path = os.path.join(dataset_path, split)
        for root, _, files in os.walk(split_path):
            for file in files:
                if file.endswith('.WRD'):
                    wrd_path = os.path.join(root, file)
                    wrd_files.append((split, wrd_path))
                    with open(wrd_path, 'r') as f:
                        for line in f:
                            _, _, label = line.strip().split()
                            word_counts[label] += 1

    # 🌟 Garder uniquement les mots dans le bon intervalle
    selected_words = {w for w, c in word_counts.items() if MIN_OCCURRENCES <= c <= MAX_OCCURRENCES}
    print(f"Nombre de mots gardés ({MIN_OCCURRENCES} ≤ occur ≤ {MAX_OCCURRENCES}): {len(selected_words)}")

    # 💾 Compteur d'occurrences extraites par mot
    extracted_counts = defaultdict(int)

    # 🔁 Deuxième passe : extraire les segments
    for split, wrd_path in tqdm(wrd_files, desc="📤 Extraction des segments"):
        wav_path = wrd_path.replace('.WRD', '.WAV')
        if not os.path.exists(wav_path):
            continue

        try:
            audio, sr = sf.read(wav_path)
        except Exception as e:
            print(f"[!] Erreur lecture {wav_path} : {e}")
            continue

        if sr != TARGET_SAMPLING_RATE:
            print(f"[!] Sample rate inattendu pour {wav_path} : {sr} Hz")
            continue

        speaker_id = os.path.basename(os.path.dirname(wrd_path))
        with open(wrd_path, 'r') as f:
            for idx, line in enumerate(f):
                try:
                    start, end, label = line.strip().split()
                    if label not in selected_words:
                        continue
                    if extracted_counts[label] >= MAX_OCCURRENCES:
                        continue

                    start, end = int(start), int(end)
                    segment = audio[start:end]
                    if len(segment) < N_FFT:
                        continue

                    label_dir = os.path.join(OUTPUT_DIR, label)
                    os.makedirs(label_dir, exist_ok=True)

                    segment_filename = f"{split}_{speaker_id}_{idx}.wav"
                    segment_path = os.path.join(label_dir, segment_filename)
                    sf.write(segment_path, segment, sr)
                    extracted_counts[label] += 1
                except Exception as e:
                    print(f"[!] Erreur extraction segment ({wrd_path}): {e}")


timit_input = "data/TIMIT/TIMIT"  # ou TEST si tu veux
extract_words_from_timit(timit_input)


