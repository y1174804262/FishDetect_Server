import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torchmetrics import F1Score
from tqdm import tqdm
def test_model(model, test_loader):
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    f1_metric = F1Score(task="binary", average='macro').to(device)
    with torch.no_grad():
        total_train_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_auc = 0
        loss_list = []
        all_preds = []
        all_labels = []
        for batch in tqdm(test_loader):
            url, certificate, label = batch
            label = label.to(device)
            pred, pred_vec = model(certificate, url)
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
            # 收集所有预测值和标签
            all_preds.extend(pred)  # 转为 numpy
            all_labels.extend(label)
            loss = model.loss_func(pred_vec, label)
            loss_list.append(loss.item())


        # 计算平均的loss值、准确率和f1值
        avg_test_loss = sum(loss_list) / len(loss_list)
        avg_test_accuracy = total_train_accuracy / len(test_loader)
        avg_test_precision = total_precision / len(test_loader)
        avg_test_recall = total_recall / len(test_loader)
        avg_test_f1 = total_f1 / len(test_loader)
        avg_test_auc = total_auc / len(test_loader)

        result = {
            'loss': avg_test_loss,
            'accuracy': avg_test_accuracy,
            'precision': avg_test_precision,
            'recall': avg_test_recall,
            'f1': avg_test_f1,
            'auc': avg_test_auc
        }

    return result
