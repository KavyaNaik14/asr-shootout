from pydub import AudioSegment
import os

input_folder = "audio"
output_folder = "fixed_audio"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):

    if file.endswith((".wav", ".ogg", ".mp3")):

        input_path = os.path.join(input_folder, file)

        output_name = os.path.splitext(file)[0] + ".wav"

        output_path = os.path.join(output_folder, output_name)

        try:

            audio = AudioSegment.from_file(input_path)

            # Convert properly
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            audio = audio.normalize()

            audio.export(
                output_path,
                format="wav"
            )

            print(f"Converted: {file}")

        except Exception as e:

            print(f"Failed: {file}")
            print(e)