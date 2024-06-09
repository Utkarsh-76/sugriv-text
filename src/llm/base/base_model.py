
import os
from src.llm.heads.text_classification_head import TextClassificationModel
from src.llm.heads.next_token_prediciton_head import NextTokenPredictionModel
from transformers import PreTrainedModel
from dotenv import load_dotenv
from torch import nn 

# Load environment variables from .env file
load_dotenv()

MAXIMUM_SEQUENCE_LENGTH = int(os.getenv("MAXIMUM_SEQUENCE_LENGTH"))
NUMBER_OF_HEADS = int(os.getenv("NUMBER_OF_HEADS"))
NUMBER_OF_LAYERS = int(os.getenv("NUMBER_OF_LAYERS"))
PAD_INDEX = int(os.getenv("PAD_INDEX"))
FEED_FORWARD_LAYER_DIMENTION = int(os.getenv("FEED_FORWARD_LAYER_DIMENTION"))
DROP_OUT_RATE = float(os.getenv("DROP_OUT_RATE"))
VOCABULARY_SIZE = os.getenv("VOCABULARY_SIZE")
D_MODEL = int(os.getenv("D_MODEL"))
BATCH_SIZE =  int(os.getenv("BATCH_SIZE"))

class TextPredictionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.text_prediction_model = NextTokenPredictionModel(PAD_INDEX,D_MODEL, MAXIMUM_SEQUENCE_LENGTH, NUMBER_OF_HEADS, NUMBER_OF_LAYERS, FEED_FORWARD_LAYER_DIMENTION, DROP_OUT_RATE,VOCABULARY_SIZE,BATCH_SIZE )
        self.text_classifiction_model = TextClassificationModel(PAD_INDEX,D_MODEL, MAXIMUM_SEQUENCE_LENGTH, NUMBER_OF_HEADS, NUMBER_OF_LAYERS, FEED_FORWARD_LAYER_DIMENTION, DROP_OUT_RATE,VOCABULARY_SIZE,BATCH_SIZE)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None ,pipeline=None ):
      # if the pipeline is text then predict the next token
      if pipeline == 'prediction':
        loss,logits= self.text_prediction_model(input_ids, attention_mask, pipeline, labels)
        return {'loss':loss,'logits':logits }
      if pipeline == 'classification':
        loss,logits= self.text_classifiction_model(input_ids, attention_mask, pipeline, labels)
        return {'loss':loss,'logits':logits }

    def top_k(self,prompt,top_k):
      return self.text_prediction_model.generate_top_k(prompt,int(top_k))
    
    def get_text_after_applying_temperature(self,prompt,temperature):
       return self.text_prediction_model.get_text_after_applying_temperature(prompt,temperature)
    
    def generate_top_p(self,prompt,top_p=1):
       return self.text_prediction_model.generate_top_p(prompt,top_p)

    def pretrain_text(self,text):
      return self.text_prediction_model.pretrain_text(text)