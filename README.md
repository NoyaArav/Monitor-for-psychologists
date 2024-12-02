# Patient AI - Monitor System for Psychologists

## Description

The **Patient Monitor System** is an innovative AI-powered platform designed to enhance psychological therapy sessions through advanced analytics and natural language processing. This system transforms traditional therapy session recordings into actionable insights, helping psychologists better understand patient progress and emotional patterns.

🎥 **Watch Demo Video** *(Link coming soon!)*

---

## Key Features

- **Automated transcription** of therapy sessions with speaker identification.
- **Real-time sentiment analysis** of patient and therapist interactions.
- **Detection and visualization** of emotional changes during sessions.
- **Semantic search** capabilities across all session transcripts.
- **Automated topic identification** and tracking.
- **Structured database storage** for easy data retrieval and analysis.

---

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/AI-Psychologist-patient-monitor-system.git
    cd patient-monitor-system
    ```

2. **Set up API keys**:
   - Create an account at [OpenAI](https://openai.com) and [AssemblyAI](https://www.assemblyai.com).
   - Add your API keys to the respective fields in `Project.py`:
     ```python
     aai.settings.api_key = "your_assemblyai_api_key"
     client = OpenAI(api_key="your_openai_api_key")
     ```

---

## How It Works

### 1. Speech-to-Text Processing
- Audio recordings are processed using AssemblyAI's API.
- Advanced speaker diarization separates patient and therapist voices.
- Timestamps and speaker labels are preserved for accurate analysis.

### 2. Sentiment Analysis & Topic Detection
- Each utterance is analyzed using OpenAI's GPT-3.5.
- Sentiment scores are assigned on a scale from -10 to 10.
- Significant emotional changes are automatically detected.
- Key discussion topics are identified and categorized.

### 3. Semantic Search
- Vector embeddings are generated for each sentence.
- Cosine similarity is used for context-aware searching.
- Search results include surrounding conversation context.

### 4. Data Visualization
- Interactive sentiment graphs show emotional patterns.
- Topic tracking across multiple sessions.
- Easy-to-read transcripts with speaker identification.

---

## Running the Project

1. **Initialize the database**:
    ```bash
    python database_handler.py
    ```

2. **Launch the Streamlit interface**:
    ```bash
    streamlit run webpage_handler.py
    ```

3. **Using the interface**:
   - Add new patients through the sidebar.
   - Upload session recordings in WAV, MP3, or MP4 format.
   - View transcripts and analysis in real-time.
   - Search across sessions using natural language queries.
   - Track emotional patterns and significant changes.

---

## Technologies Used

### Core Technologies
- **Python 3.7+**: Primary programming language.
- **OpenAI API**: Powers sentiment analysis and embedding generation.
- **AssemblyAI API**: Provides accurate speech-to-text conversion.
- **Streamlit**: Creates the interactive web interface.

### Libraries and Frameworks
- **NumPy**: Numerical computing and array operations.
- **Plotly**: Interactive data visualization.
- **SQLite3**: Local database management.
- **scikit-learn**: Machine learning utilities.

### APIs and Models
- **OpenAI's GPT-3.5**: For natural language processing.
- **OpenAI's text-embedding-3-small**: For vector embeddings.
- **AssemblyAI's speech recognition API**: With speaker diarization.

---


## Contact

For questions or support, feel free to reach out via [your-email@example.com](mailto:your-email@example.com).
