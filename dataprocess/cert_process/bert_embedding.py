"""
    词嵌入模型
"""

import torch
from transformers import BertTokenizer, BertModel

from config import bert_base_cased_path

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained(bert_base_cased_path)
model = BertModel.from_pretrained(bert_base_cased_path)
# 将模型移动到GPU
model.to(device)

def bert_embeddings(text):
    # 3. 将证书字段转换为BERT输入格式
    # BERT需要处理成tokens格式，添加特殊符号[CLS]和[SEP]
    cls_embeddings = []
    for i in range(int(len(text) / 10000) + 1):

        inputs = tokenizer(text[i*10000:(i+1)*10000], return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

        # 将输入数据移动到GPU
        inputs = {key: value.to(device) for key, value in inputs.items()}
        # 4. 获取BERT的嵌入
        # model输出是一个元组，包含了最后一层hidden states和pooler_output
        with torch.no_grad():
            outputs = model(**inputs)
        # 5. 通过 `outputs.last_hidden_state` 获取最后一层的隐藏状态（词向量表示）
        last_hidden_state = outputs.last_hidden_state
        # 6. 获取每个输入的嵌入（可以取 [CLS] token 对应的嵌入，通常用于句子级别的表示）
        # 我们选取[CLS]位置的embedding作为整个字段的嵌入
        cls_embeddings += last_hidden_state[:, 0, :]

    return cls_embeddings


if __name__ == '__main__':
    _text = "father"
    embeddings = bert_embeddings(_text)
    print(f"{_text} 的嵌入表示：{embeddings}")