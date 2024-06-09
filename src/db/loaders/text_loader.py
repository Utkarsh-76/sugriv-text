# todo make custom dataset using data loader
from datasets import load_dataset
from src.llm.tokenizers.gpt_2_tokenizer import GPTTokenizer
from torch.utils.data import DataLoader
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

VECTOR_SIZE = int(os.getenv('D_MODEL'))

class TextLoader():
    
    def __init__(self) -> None:

        # initiate the tokenizer
        self.tokenizer = GPTTokenizer().get_tokenizer()

# Tokenize the dataset
    def tokenize(self,examples):
        tokenized_texts = []
        for example in examples:
            example = example.replace("'", "\"")
            example = json.loads(example)
            input_text = example['input']
            output_text = example['output']
            full_text = input_text + ' ' + output_text
            tokenized_text = self.tokenizer(full_text, truncation=True, padding='max_length', max_length=VECTOR_SIZE)
            tokenized_texts.append(tokenized_text)
        return tokenized_texts

    
    def get_data(self,batch_size):
        ''' gets an open source dataset from hugging face currenlty requires the 'text' 
            key to be present in the dataset this should be selected via a mode flag        
        ''' 

         # split the datset into test and validation
        train_dataset = self.tokenized_datasets["test"].shuffle(seed=42).select(range(batch_size))
        valid_dataset = self.tokenized_datasets["test"].shuffle(seed=42).select(range(batch_size))

        # Then when creating your DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
        
        train_dataloader = DataLoader(train_dataloader, batch_size=batch_size)

         # return the dataset
        return {'train':train_dataloader,'valid':valid_dataloader}
    
         
         