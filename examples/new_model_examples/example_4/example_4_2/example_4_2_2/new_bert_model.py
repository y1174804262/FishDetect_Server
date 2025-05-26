from torch import nn
from transformers import AutoModel

import config


class BaseBertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.distilbert_path)
        for param in self.bert.parameters():
            param.requires_grad = False


    def forward(self, data):
        bert_out = self.bert(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask']
        )
        hidden_state = bert_out.last_hidden_state

        # 修改2：增强token特征处理
        cls_vector = hidden_state[:, 0, :]  # [batch,512]
        global_vector = hidden_state[:, :, :]  # [batch,512]

        return cls_vector, global_vector  # 返回[CLS]和增强后的token特征oken_features