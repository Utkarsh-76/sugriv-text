import os
import sys
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

window_size = 1

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
    with io.open('../data.json', 'w', encoding='utf-8') as f:
        print(all_pairs)
        f.write(json.dumps(all_pairs, ensure_ascii=False, indent=4))

if __name__ == "__main__":

    # Example text
    texts = [
        "Injection moulding uses a ram or screw-type plunger to force molten plastic or rubber material into a mould cavity;",
        "In 1846 the British inventor Charles Hancock, a relative of Thomas Hancock, patented an injection molding machine",
        "The German chemists Arthur Eichengrün and Theodore Becker invented the first soluble forms of cellulose acetate in 1903"
    ]
    
    create(texts)
    logger.info('starting finetuning data creation..')
