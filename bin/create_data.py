import os
import sys

from fastapi import Path
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import json
import re
from nltk.tokenize import word_tokenize
import  nltk
nltk.download('punkt')
import io, json
from src.utils.logger import logging as logger

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

window_size = 10

# Cleaning and tokenizing
def preprocess_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(cleaned_text)
    return tokens

all_pairs = []

# Create input-output pairs
def create_pairs(tokens, window_size):
    input_output_pairs = []
    for i in range(len(tokens) - window_size):
        input_seq = ' '.join(tokens[i:i + window_size])
        output_seq = tokens[i + window_size]
        input_output_pairs.append({'input': input_seq, 'output': output_seq})
    return input_output_pairs

def create(texts):
    print(".....................................")
    for text in texts:
        tokens = preprocess_text(text)
        pairs = create_pairs(tokens, window_size)
        all_pairs.extend(pairs)
        print(text)

# Save to JSON file
def write(file_location):
    with io.open(file_location, 'w', encoding='utf-8') as f:
        f.write(json.dumps(all_pairs, ensure_ascii=False, indent=4))