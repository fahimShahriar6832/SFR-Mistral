import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_embeddings(text_chunk: str) -> list:
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')

    # Tokenize the input text
    max_length = 4096
    inputs = tokenizer(text_chunk, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Convert embeddings to a list and return
    return embeddings.tolist()
