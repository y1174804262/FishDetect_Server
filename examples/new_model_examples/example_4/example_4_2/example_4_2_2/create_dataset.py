import pandas as pd

from examples.new_model_examples.example_2.example_2_2.example_2_2_2.cert_process import cert_resolver


def fix(url):
    # 将http://替换成https://
    if url.startswith("http://"):
        url = url.replace("http://", "https://")
    # url = url.split('://')[-1]
    # url = url.split('/')[0]
    return url

# def fix_url():
#     data = pd.read_csv("../../datasets/dataset.csv")
#     data["url"] = data["url"].apply(fix)
#     data.to_csv("../../datasets/dataset_url.csv", index=False)

if __name__ == '__main__':
    fix_url()
    exit()
def create_cert_path(id):
    path = f"D:\\yp\\dataset\\dataset_3\\certificate\\{id}.pem"
    # print(path)
    return cert_resolver(path)

def create_dataset(csv_path):
    my_collect_data = pd.read_csv(csv_path)[["id", "url", "label"]]
    my_collect_data["cert"] = my_collect_data["id"].apply(create_cert_path)
    my_collect_data.to_csv("dataset.csv", index=False)

if __name__ == '__main__':
    my_collect_csv_path = "D:\\yp\\dataset\\dataset_3\\my_collect.csv"
    create_dataset(my_collect_csv_path)