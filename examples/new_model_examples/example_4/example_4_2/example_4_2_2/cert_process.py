from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import *


# 将证书内容解析为文本进行
def cert_resolver(cert_path):
    with open(cert_path, "rb") as f:
        cert_list = f.read().split(b'\n\n')[:-1]
    pem_cert = cert_list[0]
    cert = x509.load_pem_x509_certificate(pem_cert, default_backend())
    # 仅提取()内的数据

    subject = str(cert.subject)
    data = subject[subject.find("(") + 1:subject.find(")")]
    return data



if __name__ == '__main__':
    cert_resolver("D://test//my_collect//certificate//1000004.pem")