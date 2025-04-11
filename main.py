from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from werkzeug.utils import secure_filename
import os
import logging
import sys
import io
from PyPDF2 import PdfReader
import time
import base64
import re
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Google Cloud APIs
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import storage

app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)

# Initialize Google Cloud Storage client
storage_client = storage.Client()

# Create bucket names - these should be unique across all of Google Cloud
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'book-convo-ai')
UPLOADS_BUCKET = f"{PROJECT_ID}-uploads"
BOOKS_BUCKET = f"{PROJECT_ID}-books"
AUDIO_BUCKET = f"{PROJECT_ID}-responses"

# Create buckets if they don't exist
def ensure_bucket_exists(bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        logger.info(f"Bucket {bucket_name} exists")
    except Exception as e:
        try:
            bucket = storage_client.create_bucket(bucket_name)
            logger.info(f"Created bucket {bucket_name}")
        except Exception as e:
            logger.error(f"Error creating bucket {bucket_name}: {str(e)}")
            # Fall back to temporary local storage
            return False
    return True

# Initialize buckets
for bucket_name in [UPLOADS_BUCKET, BOOKS_BUCKET, AUDIO_BUCKET]:
    ensure_bucket_exists(bucket_name)

# Create temporary local directories for processing
TEMP_FOLDER = tempfile.gettempdir()
UPLOAD_FOLDER = os.path.join(TEMP_FOLDER, 'uploads')
BOOK_FOLDER = os.path.join(TEMP_FOLDER, 'books')
AUDIO_FOLDER = os.path.join(TEMP_FOLDER, 'audio_responses')
for folder in [UPLOAD_FOLDER, BOOK_FOLDER, AUDIO_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Configure allowed file extensions
def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['wav', 'webm', 'mp3']

def allowed_book_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['pdf']

def upload_blob(bucket_name, source_file, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        if isinstance(source_file, str):  # File path
            blob.upload_from_filename(source_file)
        else:  # File-like object
            blob.upload_from_file(source_file)
            
        logger.info(f"File uploaded to {bucket_name}/{destination_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Error uploading to Cloud Storage: {str(e)}")
        return False

def download_blob(bucket_name, source_blob_name, destination_file):
    """Downloads a blob from the bucket."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        if isinstance(destination_file, str):  # File path
            blob.download_to_filename(destination_file)
        else:  # File-like object
            blob.download_to_file(destination_file)
            
        logger.info(f"Blob {source_blob_name} downloaded from {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"Error downloading from Cloud Storage: {str(e)}")
        return False

def list_blobs(bucket_name, prefix=None):
    """Lists all blobs in the bucket with the given prefix."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return list(blobs)
    except Exception as e:
        logger.error(f"Error listing blobs in {bucket_name}: {str(e)}")
        return []

def get_files():
    """Get list of uploaded books and audio recordings from Cloud Storage"""
    audio_files = []
    
    # List audio files from Cloud Storage
    try:
        audio_blobs = list_blobs(UPLOADS_BUCKET)
        for blob in audio_blobs:
            if blob.name.endswith(('.wav', '.webm', '.mp3')):
                base_name = blob.name.rsplit('.', 1)[0]
                transcript_name = base_name + '.txt'
                response_name = base_name + '_response.mp3'
                
                # Check if transcript and response exist
                transcript_exists = any(b.name == transcript_name for b in list_blobs(UPLOADS_BUCKET))
                response_exists = any(b.name == response_name for b in list_blobs(AUDIO_BUCKET))
                
                file_info = {
                    'audio': blob.name,
                    'transcript': transcript_name if transcript_exists else None,
                    'response': response_name if response_exists else None
                }
                audio_files.append(file_info)
        
        logger.info(f"Found {len(audio_files)} audio files in Cloud Storage")
    except Exception as e:
        logger.error(f"Error listing audio files from Cloud Storage: {str(e)}")
    
    # Get book files from Cloud Storage
    book_files = []
    try:
        book_blobs = list_blobs(BOOKS_BUCKET)
        for blob in book_blobs:
            if blob.name.endswith('.pdf'):
                book_info = {
                    'filename': blob.name,
                    'title': blob.name.rsplit('.', 1)[0].replace('_', ' ')
                }
                book_files.append(book_info)
        logger.info(f"Found {len(book_files)} book files in Cloud Storage")
    except Exception as e:
        logger.error(f"Error listing book files from Cloud Storage: {str(e)}")
    
    return sorted(audio_files, key=lambda x: x['audio'], reverse=True), sorted(book_files, key=lambda x: x['filename'])

def get_transcript_content(transcript_files):
    """Extract question and response content from transcript files in Cloud Storage"""
    transcript_content = {}
    response_content = {}
    
    for file_name in transcript_files:
        if file_name:
            try:
                # Download transcript file to memory
                bucket = storage_client.bucket(UPLOADS_BUCKET)
                blob = bucket.blob(file_name)
                
                if blob.exists():
                    content = blob.download_as_text()
                    parts = content.split('\n\n')
                    if len(parts) > 0:
                        transcript_content[file_name] = parts[0].replace('Question:\n', '')
                    if len(parts) > 1:
                        response_content[file_name] = parts[1].replace('Response:\n', '')
            except Exception as e:
                logger.error(f"Error reading transcript file {file_name} from Cloud Storage: {str(e)}")
                transcript_content[file_name] = "[Error reading transcript]"
                response_content[file_name] = "[Error reading response]"
    
    return transcript_content, response_content

def get_current_book():
    """Get the currently selected book from Cloud Storage"""
    try:
        # Check if current_book.txt exists in Cloud Storage
        bucket = storage_client.bucket(BOOKS_BUCKET)
        current_book_blob = bucket.blob('current_book.txt')
        
        if current_book_blob.exists():
            return current_book_blob.download_as_text().strip()
        else:
            # If no book is selected, return the first book if available
            books = list_blobs(BOOKS_BUCKET)
            pdf_books = [b.name for b in books if b.name.endswith('.pdf')]
            
            if pdf_books:
                # Save the first book as current
                current_book_blob.upload_from_string(pdf_books[0])
                return pdf_books[0]
            return None
    except Exception as e:
        logger.error(f"Error getting current book from Cloud Storage: {str(e)}")
        return None

def set_current_book(filename):
    """Set the currently selected book in Cloud Storage"""
    try:
        bucket = storage_client.bucket(BOOKS_BUCKET)
        blob = bucket.blob('current_book.txt')
        blob.upload_from_string(filename)
        logger.info(f"Current book set to {filename} in Cloud Storage")
        return True
    except Exception as e:
        logger.error(f"Error setting current book in Cloud Storage: {str(e)}")
        return False

def extract_text_from_pdf(pdf_path_or_content):
    """Extract text from a PDF file"""
    try:
        # Check if we received a path or content
        if isinstance(pdf_path_or_content, str):
            # It's a path, open the file
            with open(pdf_path_or_content, 'rb') as file:
                pdf = PdfReader(file)
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # Only add non-empty pages
                        text += page_text + "\n"
        else:
            # It's already content (bytes)
            pdf = PdfReader(pdf_path_or_content)
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Only add non-empty pages
                    text += page_text + "\n"
        
        # Remove any file paths or system-specific information
        text = re.sub(r'file:///[^\s]+', '', text)
        
        # Clean up Project Gutenberg headers and footers
        text = re.sub(r'(?i)Project Gutenberg.*?START OF (THE|THIS) (PROJECT GUTENBERG|GUTENBERG PROJECT)', 'START OF THE BOOK', text, flags=re.DOTALL)
        text = re.sub(r'(?i)End of (the |)Project Gutenberg.*', 'END OF THE BOOK', text, flags=re.DOTALL)
        
        # Basic cleanup
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        return "Error extracting text from PDF"

def transcribe_audio(file_path):
    """Process audio file with Google's Speech-to-Text API"""
    try:
        # Initialize the Speech-to-Text client
        speech_client = speech.SpeechClient()
        
        # Read audio file
        with open(file_path, 'rb') as audio_file:
            content = audio_file.read()
        
        # Configure speech recognition request
        audio = speech.RecognitionAudio(content=content)
        
        # For WEBM OPUS audio from browser, use the appropriate config
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            language_code="en-US",
            enable_automatic_punctuation=True,
            audio_channel_count=1,
        )
        
        # Perform speech recognition
        logger.info("Sending audio to Speech-to-Text API")
        response = speech_client.recognize(config=config, audio=audio)
        logger.info(f"Received response from Speech-to-Text API")
        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "
        
        return transcript.strip()
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        # For demo, if transcription fails, return a mock transcript
        logger.info("Using mock transcript since transcription failed")
        return "Tell me about this book"

def find_relevant_context(question, book_text, book_title):
    """Find relevant passages in the book related to the question"""
    question_lower = question.lower()
    
    # Extract key terms from the question
    # Remove common stop words
    stop_words = ["a", "an", "the", "is", "are", "was", "were", "be", "been", 
                  "being", "in", "on", "at", "by", "for", "with", "about", 
                  "against", "between", "into", "through", "during", "before", 
                  "after", "above", "below", "to", "from", "up", "down", "of", 
                  "off", "over", "under", "again", "further", "then", "once", 
                  "here", "there", "when", "where", "why", "how", "all", "any", 
                  "both", "each", "few", "more", "most", "other", "some", "such", 
                  "no", "nor", "not", "only", "own", "same", "so", "than", "too", 
                  "very", "s", "t", "can", "will", "just", "don", "don't", "should", 
                  "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", 
                  "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", 
                  "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", 
                  "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", 
                  "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", 
                  "won", "won't", "wouldn", "wouldn't", "tell", "me", "book", "this"]
    
    # Extract potential important words from the question
    words = re.findall(r'\b([a-zA-Z]{3,})\b', question_lower)
    search_terms = [word for word in words if word not in stop_words]
    
    # Add question-specific terms
    if "who" in question_lower or "character" in question_lower:
        search_terms.extend(["character", "protagonist", "person", "man", "woman", "boy", "girl"])
    if "where" in question_lower or "place" in question_lower or "location" in question_lower:
        search_terms.extend(["place", "location", "setting", "country", "city", "town", "house"])
    if "when" in question_lower or "time" in question_lower:
        search_terms.extend(["time", "year", "date", "century", "period", "era", "day", "night"])
    if "what" in question_lower or "plot" in question_lower or "story" in question_lower:
        search_terms.extend(["plot", "story", "narrative", "tale", "account", "happen"])
    if "theme" in question_lower or "symbol" in question_lower:
        search_terms.extend(["theme", "symbol", "meaning", "represent", "signify", "allegory"])
    
    # Remove duplicates
    search_terms = list(set(search_terms))
    
    # Split book into paragraphs or sections
    paragraphs = re.split(r'\n\s*\n|\n{2,}', book_text)
    
    # Find relevant paragraphs containing search terms
    relevant_paragraphs = []
    relevance_scores = {}
    
    for i, paragraph in enumerate(paragraphs):
        if len(paragraph.strip()) < 30:  # Skip very short paragraphs
            continue
        
        paragraph_lower = paragraph.lower()
        score = 0
        
        for term in search_terms:
            if term in paragraph_lower:
                # Weight terms that appear in the question more heavily
                term_weight = 2 if term in question_lower else 1
                score += paragraph_lower.count(term) * term_weight
        
        if score > 0:
            relevance_scores[i] = score
    
    # Get top 5 most relevant paragraphs
    top_paragraphs = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Include surrounding context if possible
    context_paragraphs = []
    for para_idx, _ in top_paragraphs:
        # Get the paragraph and potentially some surrounding context
        start_idx = max(0, para_idx - 1)
        end_idx = min(len(paragraphs) - 1, para_idx + 1)
        
        for i in range(start_idx, end_idx + 1):
            if paragraphs[i] not in context_paragraphs and len(paragraphs[i].strip()) >= 30:
                context_paragraphs.append(paragraphs[i])
    
    # If we didn't find enough relevant content, return some general information
    if not context_paragraphs:
        # Try to find a paragraph with the book title in it
        title_words = book_title.lower().replace('_', ' ').split()
        for paragraph in paragraphs:
            if any(word in paragraph.lower() for word in title_words if len(word) > 3):
                context_paragraphs.append(paragraph)
                if len(context_paragraphs) >= 3:
                    break
    
    # Still nothing? Get some paragraphs from the beginning of the book
    if not context_paragraphs:
        for paragraph in paragraphs:
            if len(paragraph.strip()) >= 50:  # Only consider substantial paragraphs
                context_paragraphs.append(paragraph)
                if len(context_paragraphs) >= 3:
                    break
    
    # Join the paragraphs with proper spacing
    context = "\n\n".join(context_paragraphs)
    
    # Limit context length to avoid overwhelming processing
    if len(context) > 3000:
        context = context[:3000] + "..."
    
    return context

def generate_answer(question, context, book_title):
    """Generate a response to the question based on the context provided"""
    try:
        question_lower = question.lower()
        
        # Get clean book title (remove file extension and underscores)
        if book_title.endswith('.pdf'):
            book_title = book_title[:-4]
        book_title = book_title.replace('_', ' ')
        
        # Check if we have context
        if not context or len(context.strip()) < 50:
            return f"I don't have enough information to answer questions about '{book_title}'. The book might not have been properly loaded or the text couldn't be extracted."
        
        # Analyze the question type to provide appropriate response
        if re.search(r'who\s+is|who\'s|character|protagonist|antagonist', question_lower):
            # Character identification questions
            # Try to identify the character being asked about
            character_match = re.search(r'who\s+is\s+([a-zA-Z\s]+)\??|about\s+([a-zA-Z\s]+)', question_lower)
            if character_match:
                character = character_match.group(1) or character_match.group(2)
                character = character.strip()
                
                # Look for sentences about this character
                sentences = re.split(r'(?<=[.!?])\s+', context)
                character_sentences = [s for s in sentences if character.lower() in s.lower()]
                
                if character_sentences:
                    # Combine a few sentences about the character
                    character_info = " ".join(character_sentences[:3])
                    return f"In '{book_title}', {character} is described as: {character_info}"
                else:
                    # Check if the character appears in the context at all
                    if character.lower() in context.lower():
                        return f"{character} appears in '{book_title}', but I couldn't find a detailed description in the passages I analyzed."
                    else:
                        return f"I couldn't find information about {character} in '{book_title}'. You might try asking about a different character or rephrasing your question."
            else:
                # General question about characters
                # Look for character descriptions
                character_indicators = ["protagonist", "main character", "hero", "heroine", "character"]
                for indicator in character_indicators:
                    pattern = re.compile(f"{indicator}\\s+is\\s+([^.!?]+)[.!?]", re.IGNORECASE)
                    match = pattern.search(context)
                    if match:
                        char_desc = match.group(1).strip()
                        return f"The {indicator} in '{book_title}' is {char_desc}."
                
                # If no specific character description found, look for character names
                names = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', context)
                common_names = {}
                for name in names:
                    if len(name) > 1:  # Skip single letters
                        common_names[name] = common_names.get(name, 0) + 1
                
                # Get the most frequently mentioned names
                top_names = sorted(common_names.items(), key=lambda x: x[1], reverse=True)[:3]
                if top_names:
                    names_list = ", ".join([name for name, _ in top_names])
                    return f"The main characters in '{book_title}' appear to include: {names_list}."
                else:
                    return f"I couldn't identify specific characters from '{book_title}' based on the content I analyzed."
                
        elif re.search(r'where|setting|place|location', question_lower):
            # Setting/location questions
            setting_indicators = ["set in", "located in", "takes place in", "setting is", "location"]
            
            for indicator in setting_indicators:
                pattern = re.compile(f"{indicator}\\s+([^.!?]+)[.!?]", re.IGNORECASE)
                match = pattern.search(context)
                if match:
                    setting = match.group(1).strip()
                    return f"'{book_title}' is {indicator} {setting}."
            
            # Look for geographic names or common setting descriptors
            locations = re.findall(r'\b(?:house|castle|manor|palace|city|town|village|forest|mountain|sea|ocean|country|kingdom|land|world)\b\s+(?:of|in)?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', context)
            if locations:
                return f"The story in '{book_title}' appears to take place in or around {', '.join(locations[:2])}."
            
            # If no specific location found, look for setting descriptions
            setting_sentences = [s for s in re.split(r'(?<=[.!?])\s+', context) if any(word in s.lower() for word in ["location", "place", "setting", "where", "country", "city", "town", "village", "land"])]
            if setting_sentences:
                return f"The setting in '{book_title}' is described as: {setting_sentences[0]}"
            
            return f"I couldn't determine the specific setting of '{book_title}' from the content I analyzed."
            
        elif re.search(r'when|time|period|era|century|year', question_lower):
            # Time period questions
            time_patterns = [
                r'(?:in|during)\s+the\s+year\s+(\d{4})',
                r'(?:in|during)\s+the\s+(\d{1,2})(?:th|st|nd|rd)\s+century',
                r'(?:in|during)\s+the\s+(medieval|renaissance|ancient|modern|victorian|elizabethan|regency|colonial|pre-historic|bronze age|iron age|stone age|ice age)',
                r'(?:in|during)\s+the\s+([A-Z][a-z]+\s+(?:era|period|age|dynasty))'
            ]
            
            for pattern in time_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    time_period = match.group(1)
                    return f"'{book_title}' is set during the {time_period} time period."
            
            # Look for any dates or years
            years = re.findall(r'\b(1[0-9]{3}|20[0-9]{2})\b', context)
            if years:
                return f"'{book_title}' appears to be set around {years[0]} or references this time period."
            
            # If no specific time period found, look for time-related descriptions
            time_sentences = [s for s in re.split(r'(?<=[.!?])\s+', context) if any(word in s.lower() for word in ["time", "period", "era", "century", "year", "decade", "age"])]
            if time_sentences:
                return f"The time period in '{book_title}' is described as: {time_sentences[0]}"
            
            return f"I couldn't determine the specific time period of '{book_title}' from the content I analyzed."
            
        elif re.search(r'what.*?(?:about|plot|story|happen|summary)', question_lower):
            # Plot/story questions
            # Try to find plot summary phrases
            plot_indicators = ["the story follows", "the plot centers", "the narrative focuses", "the tale of", "is about", "tells the story"]
            
            for indicator in plot_indicators:
                pattern = re.compile(f"{indicator}\\s+([^.!?]+(?:[.!?][^.!?]+){{0,2}})[.!?]", re.IGNORECASE)
                match = pattern.search(context)
                if match:
                    plot = match.group(1).strip()
                    return f"'{book_title}' {indicator} {plot}"
            
            # If no clear plot indicator, construct from the first few meaningful sentences
            sentences = re.split(r'(?<=[.!?])\s+', context)
            meaningful_sentences = [s for s in sentences if len(s.split()) > 5 and not re.search(r'copyright|project gutenberg|chapter|title', s, re.IGNORECASE)]
            
            if meaningful_sentences:
                summary = " ".join(meaningful_sentences[:3])
                return f"Based on the content of '{book_title}': {summary}"
            
            return f"I couldn't extract a clear plot summary for '{book_title}' from the content I analyzed."
            
        elif re.search(r'theme|symbol|meaning|message|lesson', question_lower):
            # Theme/symbol questions
            theme_indicators = ["theme", "symbolize", "represent", "message", "meaning", "signify", "allegory"]
            
            theme_sentences = [s for s in re.split(r'(?<=[.!?])\s+', context) if any(word in s.lower() for word in theme_indicators)]
            if theme_sentences:
                themes = " ".join(theme_sentences[:2])
                return f"The themes in '{book_title}' appear to involve: {themes}"
            
            # Look for common theme words
            common_themes = ["love", "death", "power", "freedom", "identity", "justice", "truth", "nature", "society", "conflict", "humanity", "time", "transformation", "good", "evil", "redemption", "sacrifice", "war", "peace"]
            found_themes = [theme for theme in common_themes if re.search(rf'\b{theme}\b', context, re.IGNORECASE)]
            
            if found_themes:
                return f"'{book_title}' appears to explore themes of {', '.join(found_themes[:3])}, based on the content I analyzed."
            
            return f"I couldn't identify specific themes in '{book_title}' from the content I analyzed."
            
        else:
            # General questions - extract the most relevant sentences
            question_keywords = [word for word in re.findall(r'\b([a-zA-Z]{4,})\b', question_lower) if word not in stop_words]
            
            relevant_sentences = []
            sentences = re.split(r'(?<=[.!?])\s+', context)
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in question_keywords):
                    relevant_sentences.append(sentence)
                    if len(relevant_sentences) >= 3:  # Limit to 3 sentences
                        break
            
            if relevant_sentences:
                answer = " ".join(relevant_sentences)
                return f"Based on '{book_title}': {answer}"
            
            # If no relevant sentences found, provide some general information
            meaningful_sentences = [s for s in sentences if len(s.split()) > 5 and not re.search(r'copyright|project gutenberg|chapter|title', s, re.IGNORECASE)]
            if meaningful_sentences:
                general_info = " ".join(meaningful_sentences[:2])
                return f"While I couldn't find specific information about your question, here's some general information from '{book_title}': {general_info}"
            
            return f"I couldn't find information to answer your question about '{book_title}'. Try asking about a different aspect of the book."
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}", exc_info=True)
        return f"I encountered an error while analyzing '{book_title}' to answer your question. The book may not have been properly processed."

def get_llm_response(question, book_text):
    """Get response from LLM based on the book content and question"""
    try:
        # Get current book title
        current_book = get_current_book() or "Unknown Book"
        book_title = current_book.rsplit('.', 1)[0].replace('_', ' ')
        
        logger.info(f"Processing question about '{book_title}': {question}")
        
        # Find relevant context from the book
        context = find_relevant_context(question, book_text, book_title)
        
        # Generate an answer based on the context
        response = generate_answer(question, context, book_title)
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}", exc_info=True)
        return f"I'm sorry, I couldn't process your question due to a technical issue. Please try again or ask a different question."

def text_to_speech(text):
    """Convert text to speech using Google's Text-to-Speech API"""
    try:
        # Initialize the Text-to-Speech client
        tts_client = texttospeech.TextToSpeechClient()
        
        # Set the input text to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", 
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        # Configure the audio settings
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Generate the speech
        logger.info("Sending text to Text-to-Speech API")
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        logger.info("Received response from Text-to-Speech API")
        
        # Return the audio content
        return response.audio_content
        
    except Exception as e:
        logger.error(f"Error processing text-to-speech: {str(e)}", exc_info=True)
        # If TTS fails, create a silent MP3 as fallback
        return create_silent_audio()

def create_silent_audio():
    """Create a silent audio file as fallback"""
    try:
        import wave
        import struct
        
        # Create a silent WAV file
        sample_rate = 8000
        duration = 2  # seconds
        
        # Generate silent audio data (all zeros)
        num_samples = int(sample_rate * duration)
        audio_data = [0] * num_samples
        
        # Convert to bytes
        byte_data = struct.pack('h' * len(audio_data), *audio_data)
        
        # Create a BytesIO object to hold the wave file
        output = io.BytesIO()
        
        # Create a wave file
        with wave.open(output, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes (16 bits)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(byte_data)
        
        # Get the wave file data
        output.seek(0)
        return output.read()
    
    except Exception as e:
        logger.error(f"Error generating simple audio: {str(e)}", exc_info=True)
        # Create an even more basic silent audio file
        return b'\x00' * 1000  # Just return some bytes to create a file

def process_question(audio_path, transcript):
    """Process a question about the book"""
    try:
        # Get the current book
        current_book = get_current_book()
        if not current_book:
            return "No book is currently selected. Please upload a book first."
        
        # Get book content from Cloud Storage
        bucket = storage_client.bucket(BOOKS_BUCKET)
        blob = bucket.blob(current_book)
        
        if not blob.exists():
            return f"Error: The selected book {current_book} could not be found in storage."
        
        # Download the book to a temporary file
        temp_book_path = os.path.join(BOOK_FOLDER, current_book)
        blob.download_to_filename(temp_book_path)
        
        # Extract text from the book
        book_text = extract_text_from_pdf(temp_book_path)
        
        # Get response from LLM
        llm_response = get_llm_response(transcript, book_text)
        logger.info(f"Generated response: {llm_response}")
        
        # Generate audio for the response using Text-to-Speech
        audio_content = text_to_speech(llm_response)
        
        # Save the response
        if audio_content:
            # Get filename components
            base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
            response_filename = base_name + "_response.mp3"
            transcript_filename = base_name + ".txt"
            
            # Save audio response to Cloud Storage
            audio_bucket = storage_client.bucket(AUDIO_BUCKET)
            audio_blob = audio_bucket.blob(response_filename)
            audio_blob.upload_from_string(audio_content, content_type='audio/mp3')
            
            # Also save locally for immediate access
            response_path = os.path.join(AUDIO_FOLDER, response_filename)
            with open(response_path, 'wb') as out:
                out.write(audio_content)
            
            # Save transcript with response to Cloud Storage
            transcript_content = f"Question:\n{transcript}\n\n" + f"Response:\n{llm_response}\n"
            transcript_bucket = storage_client.bucket(UPLOADS_BUCKET)
            transcript_blob = transcript_bucket.blob(transcript_filename)
            transcript_blob.upload_from_string(transcript_content)
            
            # Also save locally
            transcript_path = os.path.join(UPLOAD_FOLDER, transcript_filename)
            with open(transcript_path, 'w') as f:
                f.write(transcript_content)
        
        return llm_response
    
    except Exception as e:
        logger.error(f"Error in process_question: {str(e)}", exc_info=True)
        return f"I'm sorry, I encountered an error while processing your question: {str(e)}"

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return "OK", 200

@app.route('/')
def index():
    try:
        audio_files, book_files = get_files()
        current_book = get_current_book()
        
        # Get transcript content for the template
        transcript_files = [file['transcript'] for file in audio_files if file['transcript']]
        transcript_content, response_content = get_transcript_content(transcript_files)
        
        return render_template('index.html', 
                            audio_files=audio_files, 
                            book_files=book_files, 
                            current_book=current_book,
                            transcript_content=transcript_content,
                            response_content=response_content)
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
        return f"An error occurred: {str(e)}", 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio_data' not in request.files:
        flash('No audio data')
        return redirect(url_for('index'))
    
    file = request.files['audio_data']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    # Save the file
    try:
        filename = datetime.now().strftime("%Y%m%d-%I%M%S%p") + '.wav'
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        logger.info(f"Audio file saved at: {file_path}")
        
        # Upload to Cloud Storage
        upload_blob(UPLOADS_BUCKET, file_path, filename)
        
        # Process the audio
        transcript = transcribe_audio(file_path)
        logger.info(f"Transcript: {transcript}")
        
        # Process the question and get a response
        response = process_question(file_path, transcript)
        
        # Explicitly flush any file operations to ensure everything is written
        time.sleep(0.5)  # Small delay to ensure files are written
        
        flash(f"Question processed: {transcript}")
        return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"Error in upload_audio: {str(e)}", exc_info=True)
        flash(f"Error: {str(e)}")
        return redirect(url_for('index'))

@app.route('/upload_book', methods=['POST'])
def upload_book():
    if 'book_file' not in request.files:
        flash('No book file')
        return redirect(url_for('index'))
    
    file = request.files['book_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_book_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(BOOK_FOLDER, filename)
            file.save(file_path)
            logger.info(f"Book file saved at: {file_path}")
            
            # Upload to Cloud Storage
            upload_blob(BOOKS_BUCKET, file_path, filename)
            
            # Set as current book
            set_current_book(filename)
            
            flash(f"Book '{filename}' uploaded successfully")
            return redirect(url_for('index'))
        
        except Exception as e:
            logger.error(f"Error in upload_book: {str(e)}", exc_info=True)
            flash(f"Error: {str(e)}")
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a PDF file.')
    return redirect(url_for('index'))

@app.route('/select_book', methods=['POST'])
def select_book():
    book_filename = request.form.get('book_filename')
    
    # Check if the book exists in Cloud Storage
    bucket = storage_client.bucket(BOOKS_BUCKET)
    blob = bucket.blob(book_filename)
    
    if blob.exists():
        set_current_book(book_filename)
        flash(f"Selected book: {book_filename}")
    else:
        flash("Invalid book selection")
    
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files either from local temp or from Cloud Storage"""
    try:
        # First check if file exists locally
        local_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(local_path):
            return send_file(local_path)
        
        # If not found locally, try to get from Cloud Storage
        bucket = storage_client.bucket(UPLOADS_BUCKET)
        blob = bucket.blob(filename)
        
        if blob.exists():
            # Download to temp file and serve
            temp_file = os.path.join(UPLOAD_FOLDER, filename)
            blob.download_to_filename(temp_file)
            return send_file(temp_file)
        
        return "File not found", 404
    
    except Exception as e:
        logger.error(f"Error serving uploaded file: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/books/<filename>')
def book_file(filename):
    """Serve book files either from local temp or from Cloud Storage"""
    try:
        # First check if file exists locally
        local_path = os.path.join(BOOK_FOLDER, filename)
        if os.path.exists(local_path):
            return send_file(local_path)
        
        # If not found locally, try to get from Cloud Storage
        bucket = storage_client.bucket(BOOKS_BUCKET)
        blob = bucket.blob(filename)
        
        if blob.exists():
            # Download to temp file and serve
            temp_file = os.path.join(BOOK_FOLDER, filename)
            blob.download_to_filename(temp_file)
            return send_file(temp_file)
        
        return "File not found", 404
    
    except Exception as e:
        logger.error(f"Error serving book file: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/audio_responses/<filename>')
def audio_response(filename):
    """Serve audio response files either from local temp or from Cloud Storage"""
    try:
        # First check if file exists locally
        local_path = os.path.join(AUDIO_FOLDER, filename)
        if os.path.exists(local_path):
            return send_file(local_path)
        
        # If not found locally, try to get from Cloud Storage
        bucket = storage_client.bucket(AUDIO_BUCKET)
        blob = bucket.blob(filename)
        
        if blob.exists():
            # Download to temp file and serve
            temp_file = os.path.join(AUDIO_FOLDER, filename)
            blob.download_to_filename(temp_file)
            return send_file(temp_file)
        
        return "File not found", 404
    
    except Exception as e:
        logger.error(f"Error serving audio response: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/script.js')
def serve_script():
    return send_file('script.js')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)