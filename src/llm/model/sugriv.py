from src.llm.base.default_config import SugrivConfig
from src.llm.base.base_model import TextPredictionModel
from src.lib.initialize_weights import initialize_weights
import torch 

class Sugriv(torch.nn.Module):
    def __init__(self) -> None:
        super(Sugriv, self).__init__()
        self.sugriv = TextPredictionModel()
        initialize_weights(self.sugriv)

    def get_model(self):
        return self.sugriv
    
    def generate_top_k(self,prompt,top_k):
      return self.sugriv.top_k(prompt,int(top_k))
    
    def get_text_after_applying_temperature(self,prompt,temperature):
      return self.sugriv.get_text_after_applying_temperature(prompt,temperature)
    
    def generate_top_p(self,prompt,top_p):
       return self.sugriv.generate_top_p(prompt,top_p)
       
    def pretrain_text(self,text):
      return self.sugriv.pretrain_text(text)
    
    def forward(self, input_ids=None, attention_mask=None, pipeline=None, labels=None):
      return  self.sugriv.forward(input_ids, attention_mask, pipeline, labels)

# load the model
sugriv = Sugriv()