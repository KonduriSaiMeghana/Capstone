import os
import pyttsx3
import PyPDF2
import subprocess

from googletrans import Translator

# Optional: Whisper STT
import whisper

# Load Whisper model once
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

def espeak_speak(text, lang_code):
    try:
        # Construct espeak command
        command = f'espeak "{text}" -v {lang_code}'
        subprocess.run(command, shell=True)
    except Exception as e:
        print("🛑 Failed to speak:", e)

def read_pdf_lines(path):
    lines = []
    try:
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    for line in text.split('\n'):
                        lines.append(line.strip())
        return lines
    except Exception as e:
        print("🛑 Error reading PDF:", e)
        return []

def detect_blanks_and_speak(lines, lang_code="en"):
    translator = Translator()

    for line in lines:
        if "_____" in line or "______" in line or "________" in line:
            to_speak = f"Please fill the following: {line}"
        else:
            to_speak = line

        try:
            if lang_code != "en":
                translated = translator.translate(to_speak, dest=lang_code)
                to_speak = translated.text
        except:
            pass  # fallback to English if translation fails

        print(f"🔊 Speaking: {to_speak}")
        espeak_speak(to_speak, lang_code)

def list_pdfs(folder="."):
    files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    for idx, f in enumerate(files, 1):
        print(f"{idx}. {f}")
    return files

if __name__ == "__main__":
    print("📂 Available Forms:")
    forms = list_pdfs()

    if not forms:
        print("❌ No PDF forms found in current directory.")
        exit(1)

    try:
        index = int(input("\nEnter the number of the form to read: ")) - 1
        selected_file = forms[index]
    except:
        print("❌ Invalid selection.")
        exit(1)

    lang_code = input("Enter language code (e.g., en, hi, te, ta): ").strip()

    print(f"\n📄 Reading Form: {selected_file}")
    form_lines = read_pdf_lines(selected_file)
    detect_blanks_and_speak(form_lines, lang_code=lang_code)
