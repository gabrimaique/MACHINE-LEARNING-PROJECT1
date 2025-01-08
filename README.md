# MACHINE-LEARNING-PROJECT1


# README

## Project: Interview/Debate Audio Analysis

### Description
Analyze audio recordings of interviews or debates to detect media bias and impartiality. The project includes:
1. **Speaker Diarisation**: Identifies who spoke and when using Pyannote audio models.
2. **Speech-to-Text**: Generates transcripts with speaker annotations.
3. **LLM Analysis**: Extracts insights like sentiment and bias using a Large Language Model.

---

### Features
- Supports MP3 and WAV audio formats.
- Outputs include speaker timings, annotated transcripts, and advanced text analyses.
- Evaluates accuracy using Diarisation Error Rate (DER) and Word Error Rate (WER).

---

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-link>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `AudioAnalysis.ipynb` in Jupyter Notebook.

---

### Usage
1. Add your audio file (`.mp3` or `.wav`) to the specified folder.
2. Update the file path in the notebook.
3. Run the notebook to:
   - Extract speaker timings.
   - Generate transcripts.
   - Analyze transcripts for insights.

---

### Outputs
- **Diarisation Results**: Speaker time spans.
- **Transcripts**: Text with speakers identified.
- **LLM Insights**: Sentiment, bias, and more.

---

### Models Used
1. **Speaker Diarisation**: Pyannote audio models.
2. **Speech-to-Text**: Wav2Vec 2.0.
3. **LLM Analysis**: OpenAI GPT models.

---

### Notes
- Save notebook outputs before submission.
- Test with diverse audio files for better results.




