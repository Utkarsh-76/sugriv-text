from transformers import AutoTokenizer

class GPTTokenizer():
    def __init__(self):
        # use gpt tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # add the spec
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def get_tokenizer(self):
        return self.tokenizer