
import os
import sys

import torch
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import json
from src.db.loaders.text_loader import TextLoader
from torch.utils.data import Dataset
from dotenv import load_dotenv
from src.utils.logger import logging as logger
from transformers import AdamW
from src.llm.model.sugriv import sugriv
from torch.utils.data import DataLoader
from transformers import TrainingArguments

# Load environment variables from .env file
load_dotenv()

model = sugriv.get_model()
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

train_dataset = TextCompletionDataset(tokenized_datasets)
train_dataloader = DataLoader(train_dataset, batch_size=10)

# get the optimizer 
optimizer = AdamW(sugriv.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=1)


NUMBER_OF_EPOCHS = int(os.getenv("NUMBER_OF_EPOCHS"))

# declare the training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=10,
    logging_dir='./logs',
)


class Finetuner():
    def __init__(self) -> None:
        self.dataloader = train_dataloader
        self.sugriv = sugriv
        self.tokenizer = text_loader.tokenize
        self.optimizer = optimizer
        self.scheduler = scheduler


    def finetune(self):
            
        '''
        For a next token generation task, where you want to predict the next token in a sequence given the previous tokens,
        you typically generate labels by shifting the input sequence by one token.Shift the input tokens by 1 position to the right
        '''

        # Compute the number of training steps
        num_training_steps = len(self.dataloader) * training_args.num_train_epochs

        # Training loop
        progress_bar = tqdm(total=num_training_steps, desc="Training")

        average = []

        for epoch in range(training_args.num_train_epochs):

            # train the model
            sugriv.train()

            # total loss per epoch
            total_loss = 0.0

            #number of batches
            num_batches = 0

            # for each step in the train loader
            for step, batch in enumerate(self.dataloader):

                # forward pass though LLM
                outputs = self.sugriv(input_ids=batch['input_ids'].float(), labels=batch['labels'], attention_mask=batch['attention_mask'].float(),pipeline="prediction")

                # get the calculated loss
                loss = outputs['loss']

                # do a backward pass through LLM
                self.optimizer.zero_grad()
                loss.backward()

                # Update the parameters
                self.optimizer.step()

                # Update the learning rate
                self.scheduler.step()

                # calculate the toal loss
                total_loss += loss.item()

                # update the progress bar
                progress_bar.update(1)
        
        num_batches += 1
        avg_loss =  (total_loss / num_batches)
        average.append(avg_loss)
        logger.info({"loss": avg_loss})

