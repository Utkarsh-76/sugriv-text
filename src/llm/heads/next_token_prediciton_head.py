import numpy as np
import torch
import torch.nn as nn
from nltk import word_tokenize
from src.llm.transformers.text.transformer import Transformer
from src.llm.tokenizers.gpt_2_tokenizer import GPTTokenizer
from src.llm.embeddings.text_embeddings import TextEmbeddings

class NextTokenPredictionModel(nn.Module):
    def __init__(self, pad_index, d_model, max_seq_length, n_heads, n_layers, ff_dim, dropout_rate, vocab_size, batch_size):
        super(NextTokenPredictionModel, self).__init__()

        # the base model
        self.base_model = Transformer(pad_index, vocab_size, d_model, max_seq_length, n_heads, n_layers, ff_dim, dropout_rate, batch_size)

        # get the tokenizer
        self.tokenizer = GPTTokenizer()
        self.tokenizer = self.tokenizer.get_tokenizer()

        # get the embeddings
        self.embed_model = TextEmbeddings()

        self.max_seq_length = max_seq_length

        # create a linear layer
        self.linear = nn.Linear( int(d_model),int(vocab_size))  # Input size: 50258, Output size: 1024

        # Define cross-entropy loss function
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def rotate_array(self, arr, n):
        return arr[n:] + arr[:n]

    def forward(self, input_ids=None, attention_mask=None, pipeline=None, labels=None):

        # Apply linear transformation
        # Generate logits
        logits = self.base_model(input_ids)[0]

        # reconstruct the sentence from ids
        prompt_tokens = self.tokenizer.decode(input_ids[0].int(), skip_special_tokens=True)
        prompt_tokens = word_tokenize(prompt_tokens, language='english', preserve_line=True)

        # target text
        target_text = " ".join(prompt_tokens)
        target_tokens = self.tokenizer(target_text, return_tensors="pt").input_ids

        # Shift target_tokens by one to get the next token prediction target
        target_tokens_shifted = torch.roll(target_tokens, -1, dims=1)

        # Remove padding tokens from the target tensor
        target_tokens_shifted = target_tokens_shifted[:, :-1]

        # Flatten the logits and target tensors
        logits_flat = logits.view(-1, logits.size(-1))
        target_flat = target_tokens_shifted.view(-1)

        # Define cross-entropy loss criterion
        criterion = nn.CrossEntropyLoss()

        # Calculate the cross-entropy loss
        loss = criterion(logits_flat, target_flat)

        return loss.item(), logits

    def pretrain_text(self,text):

        # For the given text of length, split the text
        prompt_tokens = word_tokenize(text, language='english', preserve_line=True)

        # Create an array to store the embedding
        prompt_embedding_array = []

        # at last word and iterate through all previous
        for i, prompted_word in enumerate(prompt_tokens):

            # Preprocess the input string
            embeddings = self.embed_model.get_text_embedding(prompted_word)

            # Append the text embedding to an array
            prompt_embedding_array.append(embeddings)

        # Determine the length of the longest sequence
        max_length = max(len(seq) for seq in prompt_embedding_array)

        # Pad sequences with zeros (or another value) to make them the same length
        padded_sequences = np.array([np.pad(seq, (0, max_length - len(seq)), constant_values=50257) for seq in prompt_embedding_array])

        # Create input for the transformer
        prompt_input = torch.from_numpy(padded_sequences)
 
        # something goes wrong here
        outputs,attention = self.base_model(prompt_input)

        return outputs,attention

    def generate_top_k(self,prompt,top_k=1):
        ''' generate the next K token probalities given a prompt
            https://stackoverflow.com/questions/76397904/generate-the-probabilities-of-all-the-next-possible-word-for-a-given-text
            https://redis.io/blog/introducing-the-redis-vector-library-for-enhancing-genai-development/
        '''
        # get the output from the model
        outputs,attention = self.pretrain_text(prompt)

        # get the logits
        next_token_candidates_tensor = outputs[0][0]

        # get the top k results
        topk_candidates_indexes = torch.topk(next_token_candidates_tensor, top_k).indices.tolist()

        # Get the token probabilities for all candidates.
        all_candidates_probabilities = torch.nn.functional.softmax(next_token_candidates_tensor, dim=-1)

        # Filter the token probabilities for the top k candidates.
        topk_candidates_probabilities = all_candidates_probabilities[topk_candidates_indexes].tolist()

        # Decode the top k candidates back to words.
        topk_candidates_tokens = [self.tokenizer.decode([idx]).strip() for idx in topk_candidates_indexes]

        # Return the top k candidates and their probabilities.
        return list(zip(topk_candidates_tokens, topk_candidates_probabilities))
    
    def get_text_after_applying_temperature(self,prompt,temperature):
        ''' generate the completion using greedy algorithm using the scale provided by temperature '''

        # get the output from the model
        outputs,attention = self.pretrain_text(prompt)

        # get the logits
        logits = outputs[0][0]

        # Apply temperature scaling
        logits = logits / temperature
     
        # Softmax to get probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits))  

        # Sample from the distribution
        return np.random.choice(np.arange(len(logits)), p=probs)  
    
    def generate_top_p(self, prompt,top_p=1):
    
        # get the output from the model
        outputs,attention = self.pretrain_text(prompt)

        # get the logits
        logits = outputs[0][0]
       
        # Sort logits
        sorted_indices = np.argsort(logits) 

        # Convert sorted logits to probabilities
        sorted_probs = np.exp(logits[sorted_indices]) / np.sum(np.exp(logits))

        # Calculate the cumulative probability
        cum_probs = np.cumsum(sorted_probs) 

        # Get valid indices where cumulative probability is above threshold
        valid_indices = np.where(cum_probs >= (1 - top_p))[0]

        if len(valid_indices) > 0:
            min_valid_index = valid_indices[0]

            # Mask for valid logits
            mask = sorted_indices[min_valid_index:]
        else:
            # If no valid indices, select the last one (highest probability)
            mask = sorted_indices[-1:]

        # Randomly select an index from the valid set
        selected_index = np.random.choice(mask)

        return selected_index
