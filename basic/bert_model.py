import torch.nn as nn
from transformers import AutoModel


class IntentClassificationModel(nn.Module):
    
    def __init__(self, intent_num_labels=None,
                 model_name="bert-base-cased", dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, intent_num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs.pooler_output)

        intent_logits = self.intent_classifier(pooled_output)
        return intent_logits


class JointIntentAndSlotFillingModel(nn.Module):

    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 model=j_model, dropout_prob=0.1):
        super().__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size,
                                           intent_num_labels)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size,
                                         slot_num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        input_mask_expanded = (attention_mask.unsqueeze(-1).
                               expand(sequence_output.size()).float())
        pooled_output = torch.sum(sequence_output * input_mask_expanded, dim=1) / input_mask_expanded.sum(dim=1)
        sequence_output = self.dropout(sequence_output)
        slot_logits = self.slot_classifier(sequence_output)
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits
