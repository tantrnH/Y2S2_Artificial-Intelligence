import numpy as np
import librosa
import joblib
import os

# -------------------------------
# 设置参数
# -------------------------------
model_path = 'svm_happy_vs_sad.pkl'  # 已训练好的模型
audio_folder = 'audio'         # 存放音频的文件夹

# 特征提取函数（保持和训练时完全一致）
def extract_features(file_path, max_pad_len=200):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)

        def pad_feature(x, max_len=max_pad_len):
            if x.shape[1] < max_len:
                pad_width = max_len - x.shape[1]
                x = np.pad(x, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                x = x[:, :max_len]
            return x

        mfccs = pad_feature(mfccs)
        chroma = pad_feature(chroma)
        contrast = pad_feature(contrast)
        tonnetz = pad_feature(tonnetz)

        combined = np.vstack([mfccs, chroma, contrast, tonnetz])
        return combined.flatten()
    except Exception as e:
        print("[ERROR] Feature extraction failed:", e)
        return None

# -------------------------------
# 主流程
# -------------------------------
if __name__ == "__main__":
    print("\n[INFO] Listing available audio files...")

    if not os.path.exists(audio_folder):
        print(f"[ERROR] Folder '{audio_folder}' not found!")
        exit()

    files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))]
    if not files:
        print("[ERROR] No audio files found!")
        exit()

    for idx, file in enumerate(files):
        print(f"{idx+1}. {file}")

    selected_file = input("\nPlease type the audio file name to predict (e.g., happy01.wav): ").strip()

    file_path = os.path.join(audio_folder, selected_file)
    if not os.path.isfile(file_path):
        print(f"[ERROR] Audio file '{selected_file}' not found!")
        exit()

    print("\n[INFO] Extracting features from the audio...")
    feature = extract_features(file_path)

    if feature is None:
        print("[ERROR] Failed to extract features.")
        exit()

    feature = feature.reshape(1, -1)

    print("\n[INFO] Loading the model...")
    model = joblib.load(model_path)

    print("\n[INFO] Predicting emotion...")
    prediction = model.predict(feature)
    prediction_proba = model.decision_function(feature)

    label_mapping = {0: "Happy", 1: "Sad"}
    predicted_label = label_mapping[prediction[0]]

    confidence = np.max(prediction_proba)
    confidence_percentage = round(abs(confidence) * 100, 2)

    print("\n==============================")
    print(f"\U0001F3B5 The model predicts: {predicted_label} ({confidence_percentage}% confidence)")
    print("==============================\n")

    print("[INFO] Done!")