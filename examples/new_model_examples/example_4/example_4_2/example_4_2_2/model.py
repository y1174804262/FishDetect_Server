import torch
from torch import nn

from examples.new_model_examples.example_4.example_4_2.example_4_2_2.new_bert_model import BaseBertModel


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cert_bert = BaseBertModel()
        self.url_bert = BaseBertModel()

        # 修改3：增强跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            batch_first=True,
            dropout=0.2  # 新增dropout
        )


        # 修改5：增强分类器
        self.classifier = nn.Sequential(
            nn.Linear(768*2, 768),  # 扩大输入维度
            nn.LayerNorm(768),  # 改用LayerNorm
            nn.GELU(),
            nn.Dropout(0.4),  # 增加dropout
            nn.Linear(768, 2),
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, cert_data, url_data, labels=None):
        # 特征提取
        cert_global, cert_att = self.cert_bert(cert_data)
        url_global, url_att = self.url_bert(url_data)

        # 证书→URL
        cert_attended, _ = self.cross_attn(
            query=cert_att,
            key=url_att,
            value=url_att,
            # key_padding_mask=(url_data == 0)  # 新增mask
        )

        # URL→证书
        url_attended, _ = self.cross_attn(
            query=url_att,
            key=cert_att,
            value=cert_att,
            # key_padding_mask=(cert_data['attention_mask'] == 0)
        )

        cert_atten = torch.mean(cert_attended, dim=1)  # [batch, 768]
        url_atten = torch.mean(url_attended, dim=1)

       # 特征融合
        fused = torch.cat([cert_atten, url_atten], dim=1)  # [batch, 1024]
        # 分类
        logits = self.classifier(fused)
        pred_labels = torch.argmax(logits, dim=1)

        if labels is not None:
            loss = self.loss_func(logits, labels)
            return pred_labels, loss
        return pred_labels, logits