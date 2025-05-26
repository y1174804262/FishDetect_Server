# 模型的训练
import torch
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, optimizer):
    total_train_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_auc = 0
    loss_list = []
    # 训练模型
    model.to(device)
    # print(models)
    model.train()
    for batch in tqdm(train_loader):
        # 梯度清零
        optimizer.zero_grad()
        url, certificate, label = batch
        label = label.to(device)
        pred, loss = model(certificate, url, label)
        # 计算准确率
        accuracy = (pred == label).sum().item() / len(label)
        total_train_accuracy += accuracy

        # 计算分类指标
        pred_cpu = pred.cpu().detach().numpy()
        label_cpu = label.cpu().detach().numpy()

        # 计算精确度、召回率、F1值和AUC
        precision = precision_score(label_cpu, pred_cpu, average='binary', zero_division=0)
        recall = recall_score(label_cpu, pred_cpu, average='binary')
        f1 = f1_score(label_cpu, pred_cpu, average='binary')
        auc = roc_auc_score(label_cpu, pred_cpu)

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_auc += auc

        loss_list.append(loss.item())
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
    avg_loss = sum(loss_list) / len(loss_list)
    avg_train_accuracy = total_train_accuracy / len(train_loader)
    avg_train_precision = total_precision / len(train_loader)
    avg_train_recall = total_recall / len(train_loader)
    avg_train_f1 = total_f1 / len(train_loader)
    avg_train_auc = total_auc / len(train_loader)

    # 整理结果为字典
    train_result = {
        'loss': avg_loss,
        'accuracy': avg_train_accuracy,
        'precision': avg_train_precision,
        'recall': avg_train_recall,
        'f1': avg_train_f1,
        'auc': avg_train_auc
    }

    return train_result

