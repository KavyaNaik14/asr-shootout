from faster_whisper import WhisperModel
import pandas as pd
import os
from pydub import AudioSegment



from langdetect import detect


# LANGUAGE HANDLING SYSTEM

def detect_language(text):

    text_lower = text.lower()

    # Kannada script detection
    if any("\u0C80" <= c <= "\u0CFF" for c in text):
        return "kannada"

    # Hindi script detection
    if any(c in text for c in "ंािीुूृेैोौ"):
        return "hindi"

    # Indian Hinglish / code-mix detection
    indian_keywords = [
        "koramangala", "indiranagar", "whitefield",
        "electronic", "hsr", "btm", "jayanagar",
        "rajajinagar", "hebbal", "yelahanka",
        "sarjapur", "bellandur", "peenya"
    ]

    if any(word in text_lower for word in indian_keywords):
        return "indian_english"

    
    return "english"



# NORMALIZATION FUNCTION

def normalize_multilingual(text, lang):

    text = text.strip()

    if lang == "hindi":
        return "🇮🇳 HINDI: " + text

    elif lang == "kannada":
        return "🇮🇳 KANNADA: " + text

    elif lang == "indian_english":
        return "🇮🇳 INDIAN-ENGLISH (Hinglish): " + text

    else:
        return "ENGLISH: " + text


audio_folder = "audio"
clean_folder = "clean_audio"
output_folder = "outputs"

os.makedirs(clean_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

print("Loading Whisper model...")

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

print("Model loaded successfully!")

results = []

def clean_audio(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.normalize()
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export(output_path, format="wav")
    except Exception as e:
        print("Error:", e)

valid_extensions = (".wav", ".ogg", ".mp3", ".webm", ".flac")

for file in os.listdir(audio_folder):

    if file.lower().endswith(valid_extensions):

        print("\nProcessing:", file)

        input_path = os.path.join(audio_folder, file)

        cleaned_path = os.path.join(
            clean_folder,
            "clean_" + os.path.splitext(file)[0] + ".wav"
        )

        clean_audio(input_path, cleaned_path)

        if not os.path.exists(cleaned_path):
            continue

        segments, info = model.transcribe(
            cleaned_path,
            beam_size=5,
            language="en",
            vad_filter=True
        )

        raw_transcript = " ".join([s.text for s in segments]).strip()
        lang = detect_language(raw_transcript)
        final_transcript = normalize_multilingual(raw_transcript, lang)

        print("Transcript:", final_transcript)

        results.append({
            "audio_file": file,
            "whisper_output": final_transcript
        })

df = pd.DataFrame(results)
df.to_csv("outputs/whisper_results.csv", index=False)

print("Whisper Done")