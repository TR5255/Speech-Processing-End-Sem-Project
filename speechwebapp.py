from flask import Flask, request, jsonify, send_file
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, SpeechT5HifiGan
import torch
import re
import soundfile as sf
from pydub import AudioSegment
from transformers import SpeechT5ForTextToSpeech
# Initialize Flask app
app = Flask(__name__)

# Load Summarization Model (Local)
summarization_model_path = "C:/Users/madha/OneDrive/Desktop/SpeechWebApp/TextSummarizer"
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_path)
summarization_tokenizer = AutoTokenizer.from_pretrained(summarization_model_path)
summarizer = pipeline("summarization", model=summarization_model, tokenizer=summarization_tokenizer)

# Load TTS Model (Local)
tts_model_path = "D:/SpeechWebApp/output/content/speecht5_finetuned_madhav/checkpoint-229"
tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_path)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
tts_model.to("cpu")
vocoder.to("cpu")
# Number replacement dictionary
number_words = {i: w for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])}

def number_to_words(number):
    return number_words.get(number, str(number))

def replace_numbers_with_words(text):
    return re.sub(r'\b\d+\b', lambda x: number_to_words(int(x.group())), text)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\']', '', text)
    text = ' '.join(text.split())
    return text

def split_text(text, max_length=3):
    words = text.split()
    return [" ".join(words[i:i+max_length]) for i in range(0, len(words), max_length)]


@app.route('/')
def home():
    return "Welcome to the Speech Web App!"


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return jsonify({"summary": summary})

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data.get("text", "")
    text = replace_numbers_with_words(normalize_text(text))
    text_chunks = split_text(text)
    print("Text Chunks:", text_chunks)
    from transformers import SpeechT5Processor
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    from datasets import load_dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    speaker_embeddings = embeddings_dataset[0]["xvector"]
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

    audio_segments = []
    for idx, chunk in enumerate(text_chunks):
        inputs = processor(text=chunk, return_tensors="pt")
        print(inputs)

        speech = tts_model.generate_speech(inputs["input_ids"], vocoder=vocoder,speaker_embeddings=speaker_embeddings)
        filename = f"speech_part{idx}.wav"
        sf.write(filename, speech.numpy(), 16000)
        audio_segments.append(AudioSegment.from_wav(filename))
    
    final_audio = sum(audio_segments)
    output_file = "final_speech.wav"
    final_audio.export(output_file, format="wav")
    
    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
