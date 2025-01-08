import os
import tempfile
import torch
import numpy as np
import nltk
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from textblob import TextBlob
from collections import defaultdict
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Get Hugging Face token
hf_token = 'REPLACE WITH YOUR HUGGINGFACE TOKEN'

# Speaker Diarization
def perform_diarization(audio_file):
    # Initialize pipeline with pre-trained model
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                       use_auth_token=hf_token)

    # Move to GPU if available
    pipeline = pipeline.to(torch.device(device))

    # Apply pipeline to audio file
    diarization = pipeline(str(audio_file))

    # Extract segments by speaker
    speaker_segments = defaultdict(list)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments[speaker].append({
            'start': turn.start,
            'end': turn.end
        })

    return speaker_segments

# Function to format time in HH:MM:SS format
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

# Extract and print the speaker intervals with formatted times
def print_speaker_times(speaker_segments):
    for speaker, segments in speaker_segments.items():
        for segment in segments:
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            print(f"{speaker} {{{start_time} - {end_time}}}")


# Speech to Text for each speaker
def transcribe_segments(audio_file, speaker_segments):
    model = whisper.load_model("base").to(device)
    try:
        audio = AudioSegment.from_file(str(audio_file))
    except:
        # If format not recognized, try forcing mp3
        audio = AudioSegment.from_mp3(str(audio_file))

    speaker_transcripts = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        for speaker, segments in speaker_segments.items():
            speaker_text = ""

            for i, segment in enumerate(segments):
                start_ms = int(segment['start'] * 1000)
                end_ms = int(segment['end'] * 1000)

                segment_audio = audio[start_ms:end_ms]

                # Save segment to temporary file
                temp_path = os.path.join(temp_dir, f"segment_{i}.wav")
                segment_audio.export(temp_path, format="wav")

                # Transcribe segment
                try:
                    result = model.transcribe(temp_path)
                    speaker_text += result['text'] + " "
                except Exception as e:
                    print(f"Error transcribing segment {i} for {speaker}: {str(e)}")
                    continue

            speaker_transcripts[speaker] = speaker_text.strip()

    return speaker_transcripts

# Word Frequency and Sentiment Analysis
def analyze_text(speaker_transcripts):
    analysis_results = {}

    for speaker, text in speaker_transcripts.items():
        # Word frequency
        tokens = word_tokenize(text.lower())
        stopwords = set(nltk.corpus.stopwords.words('english'))
        tokens = [word for word in tokens if word.isalnum() and word not in stopwords]

        freq_dist = FreqDist(tokens)

        # Sentiment analysis 
        blob = TextBlob(text)
        sentiment = blob.sentiment

        analysis_results[speaker] = {
            'top_words': dict(freq_dist.most_common(10)),
            'sentiment_polarity': sentiment.polarity,
            'sentiment_subjectivity': sentiment.subjectivity
        }

    return analysis_results


def analyze_political_bias(speaker_transcripts):
    # Load sentiment analysis pipeline using RoBERTa base
    classifier = pipeline("sentiment-analysis",
                        model="cardiffnlp/twitter-roberta-base-sentiment",
                        device=0 if torch.cuda.is_available() else -1)

    # Load zero-shot classification for topic analysis
    topic_classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli",
                              device=0 if torch.cuda.is_available() else -1)

    bias_results = {}

    # Political topics to analyze
    political_topics = [
        "conservative views",
        "liberal views",
        "neutral politics",
        "economic policy",
        "social policy"
    ]

    for speaker, text in speaker_transcripts.items():
        # Analyze sentiment
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        sentiment_results = [classifier(chunk)[0] for chunk in chunks]

        # Calculate overall sentiment
        avg_sentiment = np.mean([1 if result['label'] == 'POSITIVE' else (-1 if result['label'] == 'NEGATIVE' else 0)
                               for result in sentiment_results])

        # Analyze topics
        topic_results = topic_classifier(
            text[:1024],  
            candidate_labels=political_topics,
            multi_label=True
        )
        
        bias_results[speaker] = {
            'sentiment_score': avg_sentiment,
            'top_political_topics': dict(zip(
                topic_results['labels'],
                topic_results['scores']
            )),
            'overall_tendency': 'Conservative leaning' if avg_sentiment > 0.2 else
                              'Liberal leaning' if avg_sentiment < -0.2 else
                              'Relatively neutral'
        }

    return bias_results


def setup_models():
    """Initialize all required models."""
    print("Loading models...")

    # Text generation model
    tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
    model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

    # Speech recognition
    asr = pipeline("automatic-speech-recognition",
                   model="openai/whisper-small",
                   device=0 if torch.cuda.is_available() else -1)

    # Speaker identification
    speaker_classifier = pipeline("audio-classification",
                                  model="superb/hubert-base-superb-sid",
                                  device=0 if torch.cuda.is_available() else -1)

    print("Models loaded successfully!")
    return tokenizer, model, asr, speaker_classifier

def generate_response(tokenizer, model, prompt, text):
    """Generate response from LLM."""
    input_text = f"{prompt}\n\nText: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model.generate(
        inputs.input_ids,
        max_length=150,
        num_beams=4,
        temperature=0.7,
        no_repeat_ngram_size=2
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def analyze_audio_segment(audio_path, segment_duration=30000):  # 30 seconds
    """Analyze audio in segments."""
    tokenizer, model, asr, speaker_classifier = setup_models()

    print(f"Processing audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        results = []

        for i, start in enumerate(range(0, len(audio), segment_duration)):
            print(f"\nProcessing segment {i+1}...")

            segment = audio[start:start + segment_duration]
            segment_path = os.path.join(temp_dir, f"segment_{i}.wav")
            segment.export(segment_path, format="wav")

            print("Performing speech recognition...")
            transcription = asr(segment_path)
            text = transcription["text"]

            print("Identifying speaker...")
            speaker_info = speaker_classifier(segment_path)

            prompts = {
                "political_bias": "Analyze this text for any political bias or ideological leanings. Indicate whether it is left-wing, right-wing, or centrist.",
                "main_topics": "What are the main topics or themes discussed in this text?",
                "sentiment": "What is the sentiment of this text? Consider emotions, tone, and attitude."
            }

            analysis = {
                "segment": i,
                "timestamp": f"{start//1000}-{(start+segment_duration)//1000} seconds",
                "transcription": text,
                "speaker_label": speaker_info[0]["label"],
                "speaker_confidence": speaker_info[0]["score"]
            }

            for prompt_name, prompt in prompts.items():
                print(f"Generating {prompt_name} analysis...")
                analysis[prompt_name] = generate_response(tokenizer, model, prompt, text)

            results.append(analysis)
            print(f"Completed segment {i+1}")

    return results

def print_analysis(results):
    """Print formatted analysis results."""
    for segment in results:
        print("\n" + "="*50)
        print(f"\nSegment {segment['segment']} ({segment['timestamp']})")
        print("\nTranscription:")
        print(segment['transcription'])
        print(f"\nIdentified Speaker: {segment['speaker_label']} (Confidence: {segment['speaker_confidence']:.2f})")
        print("\nPolitical Bias Analysis:")
        print(segment['political_bias'])
        print("\nMain Topics:")
        print(segment['main_topics'])
        print("\nSentiment Analysis:")
        print(segment['sentiment'])

def analyze_audio_file(audio_path):
    """Main function to analyze audio file."""
    print("Starting audio analysis...")
    results = analyze_audio_segment(audio_path)
    print("\nAnalysis complete. Printing results...")
    print_analysis(results)
    return results


def main():

    # Function to process an audio file
    def process_audio(file_path):
        print(f"Processing audio file: {file_path}\n")

        print("Performing Speaker diarization... ")
        # Perform diarization
        speaker_segments = perform_diarization(file_path)
        print(f"Found {len(speaker_segments)} speakers\n")
        print_speaker_times(speaker_segments)

        print("Generating Transcript for each speaker... ")
        # Transcribe segments
        speaker_transcripts = transcribe_segments(file_path, speaker_segments)
        
        # Print transcripts
        print("\nTranscripts:")
        for speaker, transcript in speaker_transcripts.items():
            print(f"{speaker}: {transcript}\n")

        print("Analyzing text for word frequency and sentiments of speaker...")
        # Analyze text
        analysis_results = analyze_text(speaker_transcripts)

        # Print analysis results
        for speaker, results in analysis_results.items():
            print(f"{speaker}:")
            print(f"  Top 10 words: {results['top_words']}")
            print(f"  Sentiment polarity: {results['sentiment_polarity']:.2f}")
            print(f"  Sentiment subjectivity: {results['sentiment_subjectivity']:.2f}")

        # Analyze political bias
        bias_results = analyze_political_bias(speaker_transcripts)

        # Print bias analysis results
        print("\nPolitical Bias Analysis:")
        for speaker, results in bias_results.items():
            print(f"\n{speaker}:")
            print(f"  Overall tendency: {results['overall_tendency']}")
            print(f"  Sentiment score: {results['sentiment_score']:.2f}")
            print("  Topic Analysis:")
            for topic, score in results['top_political_topics'].items():
                print(f"    {topic}: {score:.2f}")
        
        #LLM Analysis
        print("LLM Analysis")
        analyze_audio_file(file_path)


    file_path=input("Enter audio file path for analysis:")
    process_audio(file_path)
    
# Call the main function
if __name__ == "__main__":
    main()

