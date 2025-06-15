import os
import soundfile as sf
from tqdm import tqdm

TARGET_SAMPLING_RATE = 16000  # À adapter selon ton besoin
N_FFT = 512  # utilisé pour ignorer les segments trop courts
OUTPUT_DIR = 'data/segments'  # dossier où seront stockés les segments extraits

def extract_words_from_timit(dataset_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    split_dirs = ['TRAIN', 'TEST']

    for split in split_dirs:
        split_path = os.path.join(dataset_path, split)
        for root, _, files in os.walk(split_path):
            for file in files:
                if file.endswith('.WRD'):
                    wrd_path = os.path.join(root, file)
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

                    speaker_id = os.path.basename(root)
                    with open(wrd_path, 'r') as f:
                        for idx, line in enumerate(f):
                            try:
                                start, end, label = line.strip().split()
                                start, end = int(start), int(end)
                                segment = audio[start:end]

                                if len(segment) < N_FFT:
                                    continue

                                label_dir = os.path.join(OUTPUT_DIR, label)
                                os.makedirs(label_dir, exist_ok=True)

                                segment_filename = f"{split}_{speaker_id}_{idx}.wav"
                                segment_path = os.path.join(label_dir, segment_filename)

                                sf.write(segment_path, segment, sr)
                            except Exception as e:
                                print(f"[!] Erreur extraction segment: {e}")


timit_input = "data/TIMIT/TIMIT"  # ou TEST si tu veux
extract_words_from_timit(timit_input)


