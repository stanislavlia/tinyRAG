from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class Embedder():
    """Computes embedding for list of strings"""

    def __init__(self, model_name, tokenizer_name):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def _mean_pooling(self, model_output, attention_mask):
        
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        return (torch.sum(token_embeddings * input_mask_expanded, 1) / 
                        torch.clamp(input_mask_expanded.sum(1), min=1e-9))
        
    def compute_embeddings(self, sentences):
        
        encoded_input = self.tokenizer(sentences, padding=True, 
                                  truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings
        

        
        

