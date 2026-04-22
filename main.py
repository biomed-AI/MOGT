import re
import numpy as np
import pandas as pd
import random
import argparse
import os

import torch as t
import torch.nn
from torch_geometric.loader import NeighborLoader
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score, roc_auc_score

import wandb

from config import config_load
from data_preprocess_cv import get_data, DiseaseDataset,scale_data
from model import *
from utils import *

import torch.distributed as dist
import time

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_curve,matthews_corrcoef
from decimal import *

torch.cuda.set_device(4)


def arg_parse():
    parser = argparse.ArgumentParser(description="Train MOGT arguments.")
    parser.add_argument('-cv, "--cross_validation', dest="cv",
                        help="use cross validation", action="store_true")
    parser.add_argument('-g', "--gpu", dest="gpu", default=None)
    parser.add_argument('-p', "--predict", dest='pred', help="predict all nodes", action="store_true")
    parser.add_argument('-dis', "--disease", dest="disease", action='store_true')
    return parser.parse_args()


def get_model(params, dataset):
    model = MOGT(input_dim=dataset.num_node_features, hidden_dim=params['hidden_dim'], output_dim = params["out_dim"], heads=params['heads'],
                drop_rate=params['drop_rate'],  edge_dim=dataset.data.edge_dim, residual=True, devices_available=params["device"],
                num_prototypes_per_class=params["num_prototypes_per_class"])

    return model


def get_training_modules(params, dataset, pred=False):
    print("start loading model")
    loss_func = FocalLoss()

    fold = params["fold"]
    neighbors = params['neighbors']
    data = dataset.data
    model = get_model(params, dataset)

    print("start load data")
    if pred:
        loader_list = NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                input_nodes=data.train_mask[:, fold] + data.valid_mask[:, fold] + data.test_mask, shuffle=True)
        valid_loader_list, test_loader_list = None, None
    else:
        loader_list = NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                input_nodes=data.train_mask[:, fold], shuffle=True)
        valid_loader_list = NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                    input_nodes=data.valid_mask[:, fold], shuffle=True)
        test_loader_list = NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                    input_nodes=data.test_mask)
        
    unknown_loader_list = NeighborLoader(data, num_neighbors=neighbors, batch_size=params['batch_size'], directed=False,
                                    input_nodes=data.unlabeled_mask)
    optimizer = t.optim.AdamW([
        dict(params=model.convs.parameters(), weight_decay=params['weight_decay']),
        dict(params=model.lins.parameters(), weight_decay=params['weight_decay'])
    ], lr=params['lr'],weight_decay=1e-4)


    num_training_steps = sum(
        data.train_mask[:, fold]) / params['batch_size'] * params['num_epochs']
    warmup_steps = 0.1 * num_training_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    modules = {'dataset': dataset,
               'model': model,
               'loss_func': loss_func,
               'train_loader_list': loader_list,
               'valid_loader_list': valid_loader_list,
               'test_loader_list': test_loader_list,
               'unknown_loader_list': unknown_loader_list,
               'optimizer': optimizer,
               'scheduler': scheduler,
               }

    return modules


def calculate_metrics(y_true, y_pred, y_score):
    num_correct = np.equal(y_true, y_pred).sum()
    acc = (num_correct / y_true.shape[0])
    cf_matrix = confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=[0.0, 1.0])
    auprc = average_precision_score(y_true=y_true, y_score=y_score)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Youden index
    j_scores = tpr + (1 - fpr) - 1
    best_threshold = thresholds[np.argmax(j_scores)]

    return acc, cf_matrix, auprc, f1, auc,best_threshold


def pred_to_csv(configs, result,disease):
    disease = configs["disease"]
    labeled = "known" if "Label" in result.columns else "unknown"
    gene_list = get_gene_list(rename=False,disease=disease)
    gene_list['gene_index'] = np.arange(gene_list.shape[0])
    result = gene_list.merge(result)
    result = result.drop('gene_index', axis=1)
    out_dir = f"./predict/{disease}/{labeled}_result.csv"
    result.to_csv(out_dir, sep='\t', index=False)

def test(model,loader_list,devices,cutoff,fold):
    model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])
    genes = pd.DataFrame(columns=["Gene Name","gene_id"])
    steps = 0
    for data in loader_list:
        size = data.batch_size
        with torch.no_grad():
            out,sim_matrix,_ = model(data)
            out = out[:size]
        true_lab = data.y[:size][:, 1].to(devices)
        out = out.view(-1)
        pred_lab = t.zeros(size)
        pred_lab[out > cutoff] = 1
        pred_lab = pred_lab.to(devices)
        y_pred = np.append(y_pred, pred_lab.cpu().detach().numpy())
        if y_score.size == 0:
            y_score = out.cpu().detach()
        else:
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
        y_true = np.append(y_true, true_lab.cpu().detach().numpy())
        pos = data.pos[:size].to(devices).cpu().detach().numpy()
        gene = data.gene_name.loc[pos]
        genes = pd.concat([genes,gene],ignore_index=True)
    return y_true,y_pred,y_score,genes


def train(model, fold, train_loader_list, valid_loader_list, optimizer, devices, scheduler=None, loss_func=None):
    model.train()
    tot_loss = 0
    fc_loss= 0
    con_loss = 0
    acc = 0
    data = None
    out = None
    steps = 0

    for data in train_loader_list:
        size = data.batch_size
        steps = steps + 1
        optimizer.zero_grad()
        out,sim_matrix,KL_Loss = model(data)
        
        true_lab = data.y[:size][:, 1].to(devices)
        out = out[:size]
        out = out.view(-1)
        loss = loss_func(out, true_lab.float())
        alpha1 = 0.01
        alpha2 = 0.0001

        #contrastive loss
        prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, true_lab]).to(devices) 
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        positive_sim_matrix = sim_matrix * prototypes_of_correct_class
        negative_sim_matrix = sim_matrix * prototypes_of_wrong_class

        contrastive_loss = (positive_sim_matrix.sum(dim=1)) / (negative_sim_matrix.sum(dim=1))
        contrastive_loss = - torch.log(contrastive_loss).mean()

        fc_loss = fc_loss + loss
        con_loss = con_loss + contrastive_loss

        loss = loss+ alpha1 * contrastive_loss


        del out, true_lab
        loss.backward()
        tot_loss = tot_loss + loss.item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    tot_loss = tot_loss/steps
    fc_loss = fc_loss/steps
    con_loss = con_loss/steps

    model.eval()
    y_true = np.array([])
    y_score = np.array([])
    train_correct = 0
    num_train = 0
    for data in train_loader_list:
        size = data.batch_size
        with torch.no_grad():
            out,sim_matrix,_ = model(data)
            out = out[:size]
        true_lab = data.y[:size][:, 1].to(devices)
        out = out.view(-1)
        pred_lab = t.zeros(size)
        pred_lab[out > 0.5] = 1
        pred_lab = pred_lab.to(devices)
        train_correct += t.eq(pred_lab, true_lab).sum().float()
        num_train += size
        train_mask = data.train_mask[:size, fold]
        if y_score.size == 0:
            y_score = out[train_mask[:size]].cpu().detach()
        else:
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
        y_true = np.append(y_true, true_lab.cpu().detach().numpy())

    train_acc = (train_correct / num_train).cpu().detach().numpy()
    train_auprc = average_precision_score(y_true=y_true, y_score=y_score)
    train_auc = roc_auc_score(y_true, y_score)

    model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])
    genes = pd.DataFrame(columns=["gene_name","gene_id"])
    valid_loss = 0
    steps = 0
    for data in valid_loader_list:
        steps = steps + 1
        size = data.batch_size
        with torch.no_grad():
            out,sim_matrix,_ = model(data)
            out = out[:size]
        true_lab = data.y[:size][:, 1].to(devices)
        out = out.view(-1)
        pred_lab = t.zeros(size)
        pred_lab[out > 0.5] = 1
        pred_lab = pred_lab.to(devices)
        y_pred = np.append(y_pred, pred_lab.cpu().detach().numpy())
        if y_score.size == 0:
            y_score = out.cpu().detach()
        else:
            y_score = np.append(y_score, out.cpu().detach(), axis=0)
        y_true = np.append(y_true, true_lab.cpu().detach().numpy())
        valid_loss = valid_loss + loss_func(out, true_lab.float()).item()
        pos = data.pos[:size].to(devices).cpu().detach().numpy()
        gene = data.gene_name.loc[pos]
        genes = pd.concat([genes,gene],ignore_index=True)

    valid_loss = valid_loss / steps
    acc, cf_matrix, auprc, f1, auc,cutoff = calculate_metrics(y_true, y_pred, y_score)

    return tot_loss,fc_loss,con_loss ,valid_loss, acc, cf_matrix, auprc, f1, auc, y_true,y_score,genes,train_acc,train_auprc,train_auc,cutoff


def train_model(modules, params, log_name, fold, head_info=False):
    trainLoss = []
    valLoss = []
    fold = params["fold"]
    logfile = params['logfile']
    devices = params["device"]
    dataset = modules['dataset']
    model = modules['model']
    loader_list = modules['train_loader_list']
    valid_loader_list = modules['valid_loader_list']
    test_loader_list = modules["test_loader_list"]
    optimizer = modules['optimizer']
    scheduler = modules['scheduler']
    loss_func = modules['loss_func']
    disease = params["disease"]

    data = dataset[0]
    if head_info:
        config_load.print_config(logfile, params)
        with open(logfile, 'a') as f:
            print("Model: MOGT\nTrain/Valid/Test: ",
                  data.train_mask[:, fold].sum(), data.valid_mask[:, fold].sum(), data.test_mask.sum(),
                  file=f, flush=True)

    if head_info:
        with open(logfile, 'a') as f:
            print(model, file=f, flush=True)

    print('Start Training')
    vmax_auc = 0
    trigger_times = 0
    best_threshold = 0
    for epoch in range(params['num_epochs']):
        train_loss, fc_loss,con_loss,valid_loss, acc, cf_matrix, auprc, f1, auc,y_true,y_score,genes,train_acc,train_auprc,train_auc,cutoff = train(model, fold, loader_list,
                                                                                                          valid_loader_list,
                                                                                                          optimizer, devices, scheduler,
                                                                                                          loss_func)
        
        trainLoss.append(train_loss)
        valLoss.append(valid_loss)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch: {epoch}, Focal loss: {fc_loss:.4f}, contrastive loss: {con_loss:.4f}, Train loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " \
                    f"Train Auprc: {train_auprc:.4f}, Train Auroc: {train_auc:.4f}")
            print(f"Epoch: {epoch}, Valid Acc: {acc:.4f}, " \
                    f"Valid Auprc: {auprc:.4f}, Valid TP: {cf_matrix[1, 1]}, Valid F1: {f1:.4f}, Valid Auroc: {auc:.4f}")
        if epoch >= params["num_epochs"] // 10:
            if auc < vmax_auc:
                trigger_times += 1
                if trigger_times == params["num_epochs"] // 10:
                    print("Early Stopping")
                    break
            else:
                trigger_times = 0
                vmax_auc = auc
                max_epoch = epoch
                best_acc = acc
                best_tp = cf_matrix[1, 1]
                best_auprc = auprc
                best_f1 = f1
                best_threshold = cutoff
                checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}
    if not os.path.exists(os.path.join(params['out_dir'], log_name)):
        os.mkdir(os.path.join(params['out_dir'], log_name))
    if params['wandb']:
        run_name = re.findall(r'\d+', str(wandb.run.name))[0]
        model_dir = os.path.join(params['out_dir'], log_name, \
            run_name + f'_{fold}_{vmax_auc:.4f}_{best_tp}.pkl')
    else:
        model_dir = os.path.join(params['out_dir'], log_name, \
            f'{fold}_{disease}_{vmax_auc:.4f}.pkl')
    t.save(checkpoint, model_dir)

    #save valid result
    y = pd.DataFrame({'y_true': y_true, 'y_score': y_score,"cutoff":best_threshold})
    result = pd.concat([genes,y], axis=1)
    result.to_csv(params['out_dir'] + "/" +disease +"/" + disease+"_"+"valid"+"_"+str(fold)+".csv",index_label=False)

    with open(logfile, 'a') as f:
        if params['wandb']:
            print("{} epoch {}: AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f}, cutoff:{:.4f}".format(
                run_name, max_epoch, best_auprc, vmax_auc, best_acc, best_f1, best_tp, best_threshold), file=f, flush=True)
        else:
            print("epoch {}: AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f}, cutoff:{:.4f}".format(
                max_epoch, best_auprc, vmax_auc, best_acc, best_f1, best_tp, best_threshold), file=f, flush=True)
    if params['wandb']:
        wandb.finish()
    return best_auprc, vmax_auc, best_acc, best_f1, best_tp, model_dir,best_threshold

def cv_train_model(args,configs):
    disease = configs["disease"]
    dataset = get_data(configs,disease = disease)
    sum_auprc, sum_auc, sum_acc, sum_f1, sum_tp, train_result = [], [], [], [], [], []
    CV_FOLDS = configs['cv_folds']
    for i in range(CV_FOLDS):
        head_info = True if i == 0 else False 
        configs['fold'] = i
        data = dataset
        data = scale_data(data,i)
        modules = get_training_modules(configs, data)
        auprc, auc, acc, f1, tp, new_ckpt,cutoff = train_model(modules, configs, configs['log_name'], i, head_info,)
        y_score, y_pred, y_true, y_index, genes = predict(modules['model'], modules['test_loader_list'], configs, new_ckpt)
        acc, cf_matrix, auprc, f1, auc, test_cutoff = calculate_metrics(y_true, y_pred, y_score)
        #save test result
        y = pd.DataFrame({'y_true': y_true, 'y_score': y_score,"y_pred":y_pred,"cutoff":cutoff})
        result = pd.concat([genes,y], axis=1)
        result.to_csv(configs["out_dir"]+"/"+disease +"/"+disease+"_"+"test"+"_"+str(i)+".csv",index_label=False)

        tp = cf_matrix[1, 1]
        sum_auprc.append(auprc)
        sum_auc.append(auc)
        sum_acc.append(acc)
        sum_f1.append(f1)
        sum_tp.append(tp)
        with open(configs['logfile'], 'a') as f:
            print("Test AUPRC:{:.4f}, AUROC:{:.4f}, ACC:{:.4f}, F1:{:.4f}, TP:{:.1f},cutoff:{:.4f}"
                .format(auprc, auc, acc, f1, tp,test_cutoff), file=f, flush=True)
        train_result =  pred_to_df(i, train_result, y_index, y_true, y_score)
    auc_ci = calculate_confidence_interval(sum_auc)
    auprc_ci = calculate_confidence_interval(sum_auprc)
    avg_auc = sum(sum_auc) / len(sum_auc)
    avg_Auprc = sum(sum_auprc) / len(sum_auprc)
    avg_fi = sum(sum_f1) / len(sum_f1)
    with open(configs['logfile'], 'a') as f:
        print(f"{CV_FOLDS}-folds AUPRC:{avg_Auprc:.4f}+{auprc_ci}, AUROC:{avg_auc:.4f}+{auc_ci}, F1:{avg_fi:.4f}",
                file=f, flush=True)


def predict(model, loader_list, params, ckpt, labeled=True):
    devices = params["device"]
    print(f"Loading model from {ckpt} ......")
    model.load_state_dict(t.load(ckpt, map_location=model.devices_available)['state_dict'])
    model.eval()

    y_true = np.array([]) if labeled else None

    y_pred = np.array([])
    y_score = np.array([])
    y_index = np.array([])
    genes = pd.DataFrame(columns=["Gene Name","gene_id"])

    for data in loader_list:
        size = data.batch_size
        with t.no_grad():
            out,sim_matrix,_ = model(data)
            out = out[:size]
        out = t.squeeze(out)
        index = data.pos[:size]
        true_lab = data.y [:size][:, 1].to(devices) if labeled else None
        pred_lab = t.zeros(size)
        pred_lab[out > 0.5] = 1
        pred_lab = pred_lab.to(devices)
        if y_score.size == 0:
            y_score = out.cpu().detach().numpy()
        else:
            y_score = np.append(y_score, out.cpu().detach().numpy(), axis=0)
        y_pred = np.append(y_pred, pred_lab.cpu().detach().numpy())
        y_index = np.append(y_index, index.cpu().detach().numpy())

        y_true = np.append(y_true, true_lab.cpu().detach().numpy()) if labeled else None
        pos = data.pos[:size].to(devices).cpu().detach().numpy()
        gene = data.gene_name.loc[pos]
        genes = pd.concat([genes,gene],ignore_index=True)


    return y_score, y_pred, y_true, y_index,genes


def pred_to_df(i, result, y_index, y_true, y_score):
    if i == 0:
        mid = np.array([y_index, y_true, y_score]).T if y_true is not None else np.array([y_index, y_score]).T
        result = pd.DataFrame(data=mid, columns=['gene_index', 'Label',
                                    f'score_{i}'] if y_true is not None else ['gene_index',
                                    f'score_{i}'])
    else:
        mid = np.array([y_index, y_score]).T
        mid = pd.DataFrame(data=mid, columns=['gene_index', f'score_{i}'])
        result = result.merge(mid)

    return result



def predict_all(args,configs,cutoff = 0.5):
    disease = configs["disease"]
    best = configs["best"]
    dataset = get_data(configs,disease=disease)
    num_folds = 1  if best else configs["cv_folds"]
    disease = configs["disease"]
    ckpt_path = f"./outs/{disease}/"
    all_checkpoints = os.listdir(ckpt_path)
    checkpoints = [f for f in all_checkpoints if f.endswith('.pkl')]
    print(f"checkpoints:{checkpoints} " )
    known_result, unknown_result = [], []
    configs["fold"] = 1
    modules = get_training_modules(configs, dataset, pred=True)
    y_score, y_pred, y_true, y_index,_ = predict(modules['model'], modules['train_loader_list'],
                                                configs, ckpt_path + checkpoints[0])
    known_result =  pred_to_df(0, known_result, y_index, y_true, y_score)
    y_score, y_pred, y_true, y_index,_ = predict(modules['model'], modules['unknown_loader_list'],
                                                configs, ckpt_path + checkpoints[0], labeled=False)
    unknown_result = pred_to_df(0, unknown_result, y_index, y_true, y_score)
    score_col = [f"score_{i}" for i in range(num_folds)]
    known_result['avg_score'] = known_result[score_col].mean(axis=1)
    known_result['pred_label'] = known_result.apply(
        lambda x: 1 if x['avg_score'] > cutoff else 0, axis=1)
    unknown_result['avg_score'] = unknown_result[score_col].mean(axis=1)
    unknown_result['pred_label'] = unknown_result.apply(
        lambda x: 1 if x['avg_score'] > cutoff else 0, axis=1)
    print(f"候选基因的长度：{len(unknown_result)}")
    pred_to_csv(configs, known_result,disease)
    pred_to_csv(configs, unknown_result,disease)


def main(args, configs):
    if args.pred:
        predict_all(args, configs)
    else:
        configs['log_name'] = configs["disease"]
        configs['logfile'] = os.path.join(configs["log_dir"], configs["log_name"] + ".txt")
        cv_train_model(args, configs)
        



if __name__ == "__main__":
    configs = config_load.get()
    args = arg_parse()
    gpu = f"cuda:{args.gpu}" if args.gpu else 'cpu'
    configs["device"] = gpu
    main(args, configs)