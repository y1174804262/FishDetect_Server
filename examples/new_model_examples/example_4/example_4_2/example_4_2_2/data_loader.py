from torch.utils.data import Dataset
from transformers import AutoTokenizer

import config
from models.single_model.Bert_URL.data_loader import bert_load_url_feature

url_tokenizer = AutoTokenizer.from_pretrained(config.distilbert_path)
cert_tokenizer = AutoTokenizer.from_pretrained(config.distilbert_path)


class text_dataset(Dataset):
    def __init__(self, url, cert, label):
        self.url = url
        self.cert = cert
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        url_feature = bert_load_url_feature(self.url[idx], url_tokenizer)
        cert_feature = bert_load_url_feature(self.cert[idx], cert_tokenizer)

        return url_feature, cert_feature, self.label[idx]

