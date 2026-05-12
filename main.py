from deepgram import DeepgramClient, PrerecordedOptions
from pydub import AudioSegment
import pandas as pd
import os

# API KEY

DEEPGRAM_API_KEY = "6d556eaaa705247a8541d53b887583c9ef98adde"


# INIT

deepgram = DeepgramClient(DEEPGRAM_API_KEY)

audio_folder = "audio"
clean_folder = "clean_audio"
output_folder = "outputs"

os.makedirs(clean_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

results = []


# LOCALITY CORRECTIONS

corrections = {
    "Kormangala": "Koramangala",
    "Indira Nangar": "Indiranagar",
    "Wildfield": "Whitefield",
    "Raja Ji Nagar": "Rajajinagar",
    "Hember": "Hebbal",
    "Yalanka": "Yelahanka",
    "Belendur": "Bellandur",
    "Benya": "Peenya",
    "Yashwant Pura": "Yeshwanthpur",
    "Btm": "BTM",
}

# AUDIO CLEANING

def clean_audio(input_path, output_path):

    try:

        audio = AudioSegment.from_file(input_path)

        # Normalize audio
        audio = audio.normalize()

        # Convert to mono
        audio = audio.set_channels(1)

        # Set sample rate
        audio = audio.set_frame_rate(16000)

        audio.export(output_path, format="wav")

        return output_path

    except Exception as e:

        print(f"Cleaning failed: {e}")

        return input_path


# TEXT CORRECTION

def correct_text(text):

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    return text

# PROCESS FILES


for audio_file in os.listdir(audio_folder):

    if audio_file.endswith((".wav", ".ogg", ".mp3")):

        try:

            print("\n" + "=" * 50)
            print(f"Processing: {audio_file}")

            input_path = os.path.join(audio_folder, audio_file)

            cleaned_path = os.path.join(
                clean_folder,
                os.path.splitext(audio_file)[0] + "_clean.wav"
            )

           
            # CLEAN AUDIO

            final_audio = clean_audio(
                input_path,
                cleaned_path
            )

            # READ AUDIO
           

            with open(final_audio, "rb") as file:
                buffer_data = file.read()

            payload = {
                "buffer": buffer_data
            }

            
            # NOVA-3 SETTINGS
            

            options = PrerecordedOptions(
                model="nova-3",
                language="en-IN",
                smart_format=True,
                punctuate=True,
                diarize=False,
                filler_words=False
            )

           
            # TRANSCRIBE
           

            response = deepgram.listen.prerecorded.v("1").transcribe_file(
                payload,
                options
            )

            transcript = (
                response["results"]["channels"][0]
                ["alternatives"][0]["transcript"]
            )

            # Empty handling
            if transcript.strip() == "":
                transcript = "[NO SPEECH DETECTED]"

            # Apply corrections
            transcript = correct_text(transcript)

            print("Transcript:")
            print(transcript)

            results.append({
                "audio_file": audio_file,
                "transcript": transcript
            })

        except Exception as e:

            print(f"Error: {e}")

            results.append({
                "audio_file": audio_file,
                "transcript": "[ERROR]"
            })

# SAVE CSV

df = pd.DataFrame(results)

output_csv = os.path.join(
    output_folder,
    "deepgram_nova3_results.csv"
)

df.to_csv(output_csv, index=False)

print("\nDone.")
print(f"Saved results to: {output_csv}")