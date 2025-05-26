import numpy as np
import pandas as pd
import random
import torch
from sklearn.model_selection import KFold
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from examples.new_model_examples.example_4.example_4_2.example_4_2_2.data_loader import text_dataset
from examples.new_model_examples.example_4.example_4_2.example_4_2_2.model import Model
from models.CMACM.cmacm_test import test_model
from models.CMACM.cmacm_train import train_model
from models.tensor_board.my_board import writer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    dataset = pd.read_csv("../../datasets/dataset_url.csv")
    set_seed(42)

    # # # 提取1000条数据，标签分别为0和1的各有500条
    # dataset = pd.concat([
    #     dataset[dataset['label'] == 0].head(500),
    #     dataset[dataset['label'] == 1].head(500)
    # ]).reset_index(drop=True)

    # k折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 存储每一折的性能指标
    all_train_loss, all_train_f1_scores, all_train_accuracy, all_train_precision, all_train_recall, all_train_auc = [], [], [], [], [], []
    all_test_loss, all_test_f1_scores, all_test_accuracy, all_test_precision, all_test_recall, all_test_auc = [], [], [], [], [], []
    test = enumerate(kf.split(dataset))
    best_f1 = 0.0

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset, dataset['label'])):
        # set_seed(42)

        # if fold != 0:
        #     exit()
        # 分割数据集
        train_dataset = dataset.iloc[train_idx].reset_index(drop=True)
        test_dataset = dataset.iloc[val_idx].reset_index(drop=True)

        train_dataset = train_dataset.reset_index(drop=True)
        test_dataset = test_dataset.reset_index(drop=True)
        train_dataset = text_dataset(
            train_dataset['url'],
            train_dataset['cert'],
            train_dataset['label'])
        test_dataset = text_dataset(test_dataset['url'], test_dataset['cert'], test_dataset['label'])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Model().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        # optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        # optimizer = optim.SGD(model.parameters(), lr=1e-, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        epoch = 150

        # 存储当前折的训练和验证结果
        train_loss, train_accuracy, train_precision, train_recall, train_f1, train_auc = [], [], [], [], [], []
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_auc = [], [], [], [], [], []

        for i in range(epoch):
            train_result = train_model(model, train_loader, optimizer)
            for key, value in train_result.items():
                writer.add_scalar("Train/" + key + "/" + str(fold), value, i)
            print(
                f"\nTrain_Fold: {fold} Epoch {i} Loss: {train_result['loss']} Accuracy: {train_result['accuracy']} Precision: {train_result['precision']} Recall: {train_result['recall']} F1: {train_result['f1']} AUC: {train_result['auc']}")
            train_loss.append(train_result['loss']), train_accuracy.append(
                train_result['accuracy']), train_precision.append(train_result['precision']), train_recall.append(
                train_result['recall']), train_f1.append(train_result['f1']), train_auc.append(train_result['auc'])
            test_result = test_model(model, test_loader)
            for key, value in test_result.items():
                writer.add_scalar("Test/" + key + "_" + str(fold), value, i)
            print(
                f"\nTest_Fold: {fold} Epoch {i} Loss: {test_result['loss']} Accuracy: {test_result['accuracy']} Precision: {test_result['precision']} Recall: {test_result['recall']} F1: {test_result['f1']} AUC: {test_result['auc']}")
            test_loss.append(test_result['loss']), test_accuracy.append(test_result['accuracy']), test_precision.append(
                test_result['precision']), test_recall.append(test_result['recall']), test_f1.append(
                test_result['f1']), test_auc.append(test_result['auc'])
            scheduler.step(test_result['loss'])
            print(f"lr: {scheduler.get_last_lr()}")
            # # 保存f1值最优的模型
            # if test_result['f1'] > best_f1:
            #     best_f1 = test_result['f1']
            #     torch.save(model, f"models_{fold}_{i}.pth")

        all_train_loss.append(train_loss), all_train_accuracy.append(train_accuracy), all_train_precision.append(
            train_precision), all_train_recall.append(train_recall), all_train_f1_scores.append(
            train_f1), all_train_auc.append(train_auc)
        all_test_loss.append(test_loss), all_test_accuracy.append(test_accuracy), all_test_precision.append(
            test_precision), all_test_recall.append(test_recall), all_test_f1_scores.append(
            test_f1), all_test_auc.append(test_auc)

    # 计算平均指标
    avg_train_losses, avg_train_accuracy, avg_train_precision, avg_train_recall, avg_train_f1, avg_train_auc = np.mean(
        all_train_loss, axis=0), np.mean(all_train_accuracy, axis=0), np.mean(all_train_precision, axis=0), np.mean(
        all_train_recall, axis=0), np.mean(all_train_f1_scores, axis=0), np.mean(all_train_auc, axis=0)
    avg_test_losses, avg_test_accuracy, avg_test_precision, avg_test_recall, avg_test_f1, avg_test_auc = np.mean(
        all_test_loss, axis=0), np.mean(all_test_accuracy, axis=0), np.mean(all_test_precision, axis=0), np.mean(
        all_test_recall, axis=0), np.mean(all_test_f1_scores, axis=0), np.mean(all_test_auc, axis=0)

    for i in range(epoch):
        writer.add_scalar("Avg/Train/Loss", avg_train_losses[i], i)
        writer.add_scalar("Avg/Train/Accuracy", avg_train_accuracy[i], i)
        writer.add_scalar("Avg/Train/Precision", avg_train_precision[i], i)
        writer.add_scalar("Avg/Train/Recall", avg_train_recall[i], i)
        writer.add_scalar("Avg/Train/F1", avg_train_f1[i], i)
        writer.add_scalar("Avg/Train/AUC", avg_train_auc[i], i)
        writer.add_scalar("Avg/Test/Loss", avg_test_losses[i], i)
        writer.add_scalar("Avg/Test/Accuracy", avg_test_accuracy[i], i)
        writer.add_scalar("Avg/Test/Precision", avg_test_precision[i], i)
        writer.add_scalar("Avg/Test/Recall", avg_test_recall[i], i)
        writer.add_scalar("Avg/Test/F1", avg_test_f1[i], i)
        writer.add_scalar("Avg/Test/AUC", avg_test_auc[i], i)

        # break

        # 计算平均指标

    avg_f1 = sum(avg_test_f1) / len(avg_test_f1)
    avg_accuracy = sum(avg_test_accuracy) / len(avg_test_accuracy)
    avg_precision = sum(avg_test_precision) / len(avg_test_precision)
    avg_recall = sum(avg_test_recall) / len(avg_test_recall)
    avg_auc = sum(avg_test_auc) / len(avg_test_auc)

    print(
        f"测试集上平均 F1: {avg_f1} 平均准确率: {avg_accuracy} 平均精确度: {avg_precision} 平均召回率: {avg_recall} 平均AUC: {avg_auc}")

    writer.close()