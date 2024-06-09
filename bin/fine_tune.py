
import os
import sys

import torch
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import json
from src.db.loaders.text_loader import TextLoader
from torch.utils.data import Dataset
from dotenv import load_dotenv
from transformers import AdamW
from src.llm.model.sugriv import sugriv
from torch.utils.data import DataLoader

# Load environment variables from .env file
load_dotenv()

sugriv = sugriv.get_model()
text_loader = TextLoader()

# Load the dataset
with open('../data.json', 'r') as f:
    dataset = json.load(f)

# Tokenize the dataset
def collect(item):
    return str(item)

dataset = [collect(record) for record in dataset]
tokenized_datasets = text_loader.tokenize(dataset)

# Define the dataset class
class TextCompletionDataset(Dataset):
    def __init__(self, tokenized_texts):
        self.input_ids = [data['input_ids'] for data in tokenized_texts]
        self.attention_mask = [data['attention_mask'] for data in tokenized_texts]
        self.labels = [data['input_ids'] for data in tokenized_texts]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx]),
            'labels': torch.tensor(self.labels[idx])
        }
    
NUMBER_OF_EPOCHS = int(os.getenv("NUMBER_OF_EPOCHS"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))

train_dataset = TextCompletionDataset(tokenized_datasets)
train_dataloader = DataLoader(train_dataset, batch_size=10)

# get the optimizer 
optimizer = AdamW(sugriv.parameters(), lr=2e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=1)

class Finetuner():
    def __init__(self) -> None:
        self.dataloader = train_dataloader
        self.sugriv = sugriv
        self.tokenizer = text_loader.tokenize
        self.optimizer = optimizer
        self.scheduler = scheduler

    def find_average(self,arr):
        total = sum(arr)
        num_elements = len(arr)
        if num_elements == 0:
            return 0  # to handle division by zero
        average = total / num_elements
        return average

    def finetune(self):
        '''
        For a next token generation task, where you want to predict the next token in a sequence given the previous tokens,
        you typically generate labels by shifting the input sequence by one token. Shift the input tokens by 1 position to the right.
        '''
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        sugriv.to(device)

        for epoch in range(20):
            sugriv.train()
            total_loss = 0  # Initialize total loss for the epoch
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                # forward pass though LLM
                outputs = self.sugriv(input_ids=batch['input_ids'].float(), labels=batch['labels'], attention_mask=batch['attention_mask'].float(),pipeline="prediction")

                # get the calculated loss
                loss = outputs['loss']

                self.optimizer.zero_grad()
                
                loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
           
                total_loss += loss.item()  # Accumulate the loss

            if total_loss != 0:
                mean_loss = total_loss / len(train_dataloader)  # Calculate mean loss for the epoch
                print(f"Epoch: {epoch + 1}, Mean Loss: {mean_loss}")


Finetuner().finetune()