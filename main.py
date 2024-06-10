#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function
from sklearn.model_selection import KFold
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from utils.dataset import GraphDataset
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import Trainer, Evaluator, collate
from option import Options

from sklearn.metrics import precision_score, recall_score, f1_score

from models.GraphTransformer import Classifier
from models.weight_init import weight_init

args = Options().parse()
n_class = args.n_class

torch.cuda.synchronize()
torch.backends.cudnn.deterministic = True

data_path = args.data_path
model_path = args.model_path
if not os.path.isdir(model_path): os.mkdir(model_path)
log_path = args.log_path
if not os.path.isdir(log_path): os.mkdir(log_path)
task_name = args.task_name

print(task_name)
###################################
train = args.train
test = args.test
graphcam = args.graphcam
print("train:", train, "test:", test, "graphcam:", graphcam)

##### Load datasets
print("preparing datasets and dataloaders......")
batch_size = args.batch_size

if train:
    ids_train = open(args.train_set).readlines()
    dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train)
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
    total_train_num = len(dataloader_train) * batch_size

ids_val = open(args.val_set).readlines()
dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val)
dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=False, pin_memory=True)
total_val_num = len(dataloader_val) * batch_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##### creating models #############
num_splits = 5
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(kf.split(ids_train)):
    print(f"Fold {fold + 1}")

    dataset_train_fold = GraphDataset(os.path.join(data_path, ""), [ids_train[i] for i in train_index])
    dataloader_train_fold = torch.utils.data.DataLoader(dataset=dataset_train_fold, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
    total_train_num_fold = len(dataloader_train_fold) * batch_size

    dataset_val_fold = GraphDataset(os.path.join(data_path, ""), [ids_train[i] for i in val_index])
    dataloader_val_fold = torch.utils.data.DataLoader(dataset=dataset_val_fold, batch_size=batch_size, num_workers=10, collate_fn=collate, shuffle=False, pin_memory=True)
    total_val_num_fold = len(dataloader_val_fold) * batch_size
print("creating models......")

num_epochs = args.num_epochs
learning_rate = args.lr

model = Classifier(n_class)
model = nn.DataParallel(model)
if args.resume:
    print('load model{}'.format(args.resume))
    model.load_state_dict(torch.load(args.resume))

if torch.cuda.is_available():
    model = model.cuda()
#model.apply(weight_init)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)       # best:5e-4, 4e-3
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100], gamma=0.1) # gamma=0.3  # 30,90,130 # 20,90,130 -> 150

##################################

criterion = nn.CrossEntropyLoss()

if not test:
    writer = SummaryWriter(log_dir=log_path + task_name)
    f_log = open(log_path + task_name + ".log", 'w')

trainer = Trainer(n_class)
evaluator = Evaluator(n_class)

best_pred = 0.0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.
    total = 0.

    current_lr = optimizer.param_groups[0]['lr']
    print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch + 1, current_lr, best_pred))

    all_labels = []
    all_preds = []

    if train:
        for i_batch, sample_batched in enumerate(dataloader_train):
            scheduler.step(epoch)

            preds, labels, loss = trainer.train(sample_batched, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += len(labels)

            trainer.metrics.update(labels, preds)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            if (i_batch + 1) % args.log_interval_local == 0:
                print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total, total_train_num, train_loss / total, trainer.get_scores()))
                trainer.plot_cm()

    if not test:
        print("[%d/%d] train loss: %.3f; agg acc: %.3f" % (total_train_num, total_train_num, train_loss / total, trainer.get_scores()))
        trainer.plot_cm()

    if epoch % 1 == 0:
        with torch.no_grad():
            model.eval()
            print("evaluating...")

            total = 0.
            batch_idx = 0

            val_labels = []
            val_preds = []

            for i_batch, sample_batched in enumerate(dataloader_val):
                preds, labels, _ = evaluator.eval_test(sample_batched, model, graphcam)

                total += len(labels)

                evaluator.metrics.update(labels, preds)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

                if (i_batch + 1) % args.log_interval_local == 0:
                    print('[%d/%d] val agg acc: %.3f' % (total, total_val_num, evaluator.get_scores()))
                    evaluator.plot_cm()

            val_precision = precision_score(val_labels, val_preds, average='macro')
            val_recall = recall_score(val_labels, val_preds, average='macro')
            val_f1 = f1_score(val_labels, val_preds, average='macro')

            print('[%d/%d] val agg acc: %.3f; val precision: %.3f; val recall: %.3f; val F1: %.3f' % (total_val_num, total_val_num, evaluator.get_scores(), val_precision, val_recall, val_f1))
            evaluator.plot_cm()

            # torch.cuda.empty_cache()

            val_acc = evaluator.get_scores()
            if val_acc > best_pred:
                best_pred = val_acc
                if not test:
                    print("saving model...")
                    torch.save(model.state_dict(), model_path + task_name + ".pth")

            log = ""
            log = log + 'epoch [{}/{}] ------ acc: train = {:.4f}, val = {:.4f}, precision = {:.4f}, recall = {:.4f}, F1 = {:.4f}'.format(epoch + 1, num_epochs, trainer.get_scores(), evaluator.get_scores(), val_precision, val_recall, val_f1) + "\n"

            log += "================================\n"
            print(log)
            if test: break

            f_log.write(log)
            f_log.flush()

            writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()}, epoch + 1)
            writer.add_scalars('precision', {'val precision': val_precision}, epoch + 1)
            writer.add_scalars('recall', {'val recall': val_recall}, epoch + 1)
            writer.add_scalars('F1', {'val F1': val_f1}, epoch + 1)           

    trainer.reset_metrics()
    evaluator.reset_metrics()
if not test: f_log.close()
