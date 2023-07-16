import torch.nn as nn
from transformers import AutoModel

class IntentClassificationModel(nn.Module):
    
    def __init__(self, intent_num_labels=None, model_name="bert-base-cased", dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)

        # Use the default linear activation (no softmax) to compute logits.
        # The softmax normalization will be computed in the loss function
        # instead of the model itself.
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, intent_num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)

        intent_logits = self.intent_classifier(pooled_output)
        return intent_logits