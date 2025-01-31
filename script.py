import openai
import argparse
import os

# Set your OpenAI API key

openai.api_key = os.getenv("OPENAI_API_KEY")

# Argument parser for handling CLI inputs
parser = argparse.ArgumentParser(description="Transcribe and summarize multiple audio files.")
parser.add_argument("filenames", nargs='+', type=str, help="List of audio files to process")
parser.add_argument("prompt", type=str, help="Prompt for summarization")
args = parser.parse_args()

# Ensure necessary directories exist
if not os.path.exists("Transcriptions"):
    os.makedirs("Transcriptions")

if not os.path.exists("Summaries"):
    os.makedirs("Summaries")

for filename in args.filenames:
    try:
        full_filename = f"./Recordings/{filename}"
        
        # Step 1: Open the audio file and transcribe it
        print(f"Opening and transcribing the audio file: {filename}...")
        with open(full_filename, "rb") as audio_file:
            transcription_response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        
        transcription = transcription_response['text']
        transcription_file_name = f"transcription_{filename}.txt"
        transcription_file_path = f"Transcriptions/{transcription_file_name}"

        # Save the transcription
        with open(transcription_file_path, "w") as transcription_file:
            transcription_file.write(transcription)
        print(f"Transcription saved for {filename}.")

        # Step 2: Summarize the transcription
        print(f"Generating summary for {filename}...")
        prompt = f"{args.prompt} \n Transcription: {transcription}"

        summary_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional notetaker with 20 years of experience taking notes and incorporating complex ideas and topics."},
                {"role": "user", "content": prompt}
            ]
        )
        summary_text = summary_response['choices'][0]['message']['content']

        # Save the summary
        summary_filename = f"summary_{filename}.txt"
        summary_path = f"Summaries/{summary_filename}"
        with open(summary_path, "w") as summary_file:
            summary_file.write("Summary:\n")
            summary_file.write(summary_text + "\n\n")
        
        print(f"Summary saved for {filename}.")

    except openai.error.OpenAIError as e:
        print(f"OpenAI API error for {filename}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")

print("All files processed successfully.")
