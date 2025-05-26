import torch
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class URLDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,    # 限制最大长度
            padding='max_length',       # 填充到最大长度
            truncation=True,            # 截断到最大长度
            return_tensors="pt"         # 返回 PyTorch 张量
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),  # 注意力掩码
            'label': torch.tensor(label, dtype=torch.long)
        }


def bert_load_url_feature(url, tokenizer):
    encoding = tokenizer(
        str(url),
        max_length=128,  # 限制最大长度
        padding='max_length',  # 填充到最大长度
        truncation=True,  # 截断到最大长度
        return_tensors="pt"  # 返回 PyTorch 张量
    ).to(device)

    url_feature = {
        'input_ids': encoding['input_ids'].squeeze(0),
        'attention_mask': encoding['attention_mask'].squeeze(0)  # 注意力掩码
    }
    return url_feature