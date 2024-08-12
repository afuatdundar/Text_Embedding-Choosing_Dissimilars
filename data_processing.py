import os
import re
import string
import pandas as pd
import nltk
from nltk import word_tokenize
from langdetect import detect, DetectorFactory
from googletrans import Translator

# NLTK punkt verisini indirme
nltk.download('punkt')

# Langdetect'in rastgelelik sorunlarını önlemek için sabitleme
DetectorFactory.seed = 0

def is_numeric(text):
    """Check if the text is numeric only."""
    return text.strip().isdigit()

def is_url(text):
    """Check if the text contains only URLs."""
    url_pattern = re.compile(r'(https?://\S+)')
    return bool(url_pattern.fullmatch(text.strip()))

def read_texts(directory):
    docs = [f for f in os.listdir(directory) if f.endswith(".txt")]
    translator = Translator()
    data = []

    for file_name in docs:
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()  # Ensure text is stripped of extra whitespace
                
                # Check if text is numeric-only or URL-only
                if is_numeric(text) or is_url(text):
                    data.append({'file_name': file_name, 'text': text})
                else:
                    # Detect and translate if not numeric-only or URL-only
                    try:
                        detected_language = detect(text)
                        if detected_language != 'en':
                            text = translator.translate(text, src=detected_language, dest='en').text
                    except Exception as e:
                        print(f"Language detection or translation failed for file {file_name}: {e}")
                        # In case of failure, include the original text
                        data.append({'file_name': file_name, 'text': text})
                        continue
                    data.append({'file_name': file_name, 'text': text})
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return data

def remove_punctuation(doc):
    text = word_tokenize(doc)
    text = [w for w in text if w not in string.punctuation]
    return " ".join(text)

def remove_emoji(text):
    return re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001FB00-\U0001FBFF\U0001FE00-\U0001FE0F\U0001F004]+',
        '', text)

def preprocess_texts(data):
    df = pd.DataFrame(data)
    
    # Debugging: Print DataFrame columns and a few rows
    print("DataFrame columns:", df.columns)
    print("First few rows:\n", df.head())
    
    # Check if 'text' column exists
    if 'text' not in df.columns:
        raise ValueError("The DataFrame does not contain 'text' column.")
    
    # Process texts
    df["cleaned_text"] = df["text"].apply(lambda row: remove_punctuation(remove_emoji(row.lower())) if row else "")
    return df["cleaned_text"].tolist()
