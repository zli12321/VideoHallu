from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import save_file, load_file
import torch
import torch.nn as nn
import os

device = "cuda:7"

class BertAnswerScorer(nn.Module):
    """
    A simple BERT-based regressor to predict a single score in [0, 1].
    """
    def __init__(self, model_name="answerdotai/ModernBERT-base", device=device):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name).to(device)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(p=0.1)
        self.regressor = nn.Linear(hidden_size, 1).to(device)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is not None and getattr(self.bert.config, "type_vocab_size", 1) > 1:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        # Use pooler_output if available; otherwise, take the CLS token representation.
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        score = torch.sigmoid(logits).squeeze(-1)
        return score

def load_model_and_tokenizer_from_folder(checkpoint_folder, base_model_name="answerdotai/ModernBERT-base", device=device):
    """
    Loads the model and tokenizer from the given checkpoint folder.
    """

    print('Device: ', device)
    if device is None:
        device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)
    config = AutoConfig.from_pretrained(checkpoint_folder)
    
    model = BertAnswerScorer(base_model_name)
    
    safe_path = os.path.join(checkpoint_folder, "pytorch_model.safetensors")
    state_dict = load_file(safe_path)
    model.load_state_dict(state_dict)
    # print('Model: ', model)
    
    model.to(device)
    model.eval()
    
    return model, tokenizer

def get_score(model, tokenizer, reference, generated_response, device=device):
    MAX_LENGTH = 2048
    if device is None:
        device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    combined_text = f"{reference} [SEP] {generated_response}"
    
    encoding = tokenizer(
        combined_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    
    for key in encoding:
        encoding[key] = encoding[key].to(device)
    
    model.eval()
    with torch.no_grad():
        normalized_score = model(**encoding)
    
    # Map normalized score to the [1, 5] scale.
    final_score = 1.0 + 4.0 * normalized_score.item()
    return normalized_score.item(), final_score



class RewardBert:
    def __init__(self):
        checkpoint_folder = os.path.join(
            "/home/ubuntu/Illinois",
            "checkpoint-epoch-3"
        )
        self.model, self.tokenizer = load_model_and_tokenizer_from_folder(
            checkpoint_folder, base_model_name="answerdotai/ModernBERT-base"
        )
    
    def compute_score(self, extracted_answer, label):
        # This returns a tuple (normalized_score, final_score)
        return get_score(self.model, self.tokenizer, label, extracted_answer)




bert = RewardBert()

print(bert.compute_score('a', 'b'))