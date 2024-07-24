import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score as F1
from sklearn.metrics import roc_auc_score as auc
from scipy.optimize import linear_sum_assignment
from collections import Counter

best_result = {}
best_result["known_acc"], best_result["novel_acc"], best_result["all_acc"] = 0, 0, 0
def Valid_train(model, data, valid_batch_size, device):

    data.batch_size = valid_batch_size

    label_t = []
    label_p = []
    unlabel_t = []
    unlabel_p1 = []

    with torch.no_grad():
        for batch_encode_input, _, _, batch_seq_length, labels, unknow_labels in data.iterate_train_data():
            batch_encode_input, batch_seq_length, known_labels, unknow_labels = batch_encode_input[:2*data.batch_size].to(device), batch_seq_length[:2*data.batch_size].to(
                device), labels.to(device), unknow_labels.to(device)


            feats, outputs = model(batch_encode_input, batch_seq_length)

            known_preds = outputs[:data.batch_size].max(dim=-1)[1]

            for item in known_labels.cpu().numpy():
                label_t.append(item)
            for item in known_preds.cpu().numpy():
                label_p.append(item)
            unknown_preds = outputs[data.batch_size:2*data.batch_size].max(dim=-1)[1]
            for item in unknow_labels.cpu().numpy():
                unlabel_t.append(item)
            for item in unknown_preds.cpu().numpy():
                unlabel_p1.append(item)




    all_t = np.array(label_t + unlabel_t).reshape(-1)
    all_p1 = np.array(label_p + unlabel_p1).reshape(-1)
    label_t = np.array(label_t).reshape(-1)
    label_p = np.array(label_p).reshape(-1)
    unlabel_t = np.array(unlabel_t).reshape(-1)
    unlabel_p1 = np.array(unlabel_p1).reshape(-1)

    know_result = hungarian_evaluate(torch.from_numpy(label_t), torch.from_numpy(label_p))
    unknow_result = hungarian_evaluate(torch.from_numpy(unlabel_t), torch.from_numpy(unlabel_p1), offset=2)
    all_result = hungarian_evaluate(torch.from_numpy(all_t), torch.from_numpy(all_p1))
    known_print = 'known acc: {:.1f}, known nmi: {:.4f}, known ari: {:.4f}'.format(know_result['acc'],
                                                                                   know_result['nmi'],
                                                                                   know_result['ari'])
    novel_print = 'novel acc: {:.1f}, novel nmi: {:.4f}, unknown ari: {:.4f}'.format(unknow_result['acc'],
                                                                                       unknow_result['nmi'],
                                                                                       unknow_result['ari'])
    all_print = 'all acc: {:.1f}, all nmi: {:.4f}, all ari: {:.4f}'.format(all_result['acc'],
                                                                           all_result['nmi'],
                                                                           all_result['ari'])
    print(known_print, end=' | ')
    print(novel_print, end=' | ')
    print(all_print)

def Valid_val(model, data, valid_batch_size, device, num_labels, dataset, novel_map, novel_type):
    data.batch_size = valid_batch_size
    all_t = []
    all_p1 = []
    label_t = []
    label_p = []
    unlabel_t = []
    unlabel_p1 = []

    with torch.no_grad():
        for batch_encode_input, _, _, batch_seq_length, labels in data.iterate_val_data():
            batch_encode_input = torch.LongTensor(batch_encode_input).to(device)
            batch_seq_length = torch.LongTensor(batch_seq_length).to(device)
            labels = torch.LongTensor(labels).to(device)

            # forward
            # forward
            feats, outputs = model(batch_encode_input, batch_seq_length)

            preds = outputs.max(dim=-1)[1]
            mask_lab = labels < num_labels
            for item in labels[mask_lab].cpu().numpy():
                label_t.append(item)
            for item in preds[mask_lab].cpu().numpy():
                label_p.append(item)

            mask_lab = labels >= num_labels
            for item in labels[mask_lab].cpu().numpy():
                unlabel_t.append(item)
            for item in preds[mask_lab].cpu().numpy():
                unlabel_p1.append(item)

            all_t.append(labels.cpu().numpy())
            all_p1.append(preds.cpu().numpy())

        all_t = np.array(all_t).reshape(-1)
        all_p1 = np.array(all_p1).reshape(-1)
        label_t = np.array(label_t).reshape(-1)
        label_p = np.array(label_p).reshape(-1)
        unlabel_t = np.array(unlabel_t).reshape(-1)
        unlabel_p1 = np.array(unlabel_p1).reshape(-1)

        know_result = hungarian_evaluate(torch.from_numpy(label_t), torch.from_numpy(label_p))
        unknow_result = hungarian_evaluate(torch.from_numpy(unlabel_t), torch.from_numpy(unlabel_p1), offset=2)
        all_result = hungarian_evaluate(torch.from_numpy(all_t), torch.from_numpy(all_p1))
        known_print = 'known acc: {:.1f}, known nmi: {:.4f}, known ari: {:.4f}'.format(know_result['acc'],
                                                                                       know_result['nmi'],
                                                                                       know_result['ari'])
        novel_print = 'novel acc: {:.1f}, novel nmi: {:.4f}, unknown ari: {:.4f}'.format(unknow_result['acc'],
                                                                                         unknow_result['nmi'],
                                                                                         unknow_result['ari'])
        all_print = 'all acc: {:.1f}, all nmi: {:.4f}, all ari: {:.4f}'.format(all_result['acc'],
                                                                               all_result['nmi'],
                                                                               all_result['ari'])
        print(known_print, end=' | ')
        print(novel_print, end=' | ')
        print(all_print)

        if best_result["all_acc"] < all_result['acc']:
            best_result["known_acc"], best_result["novel_acc"], best_result[
                "all_acc"] = know_result['acc'], unknow_result['acc'], all_result['acc']
            # save_model_filename = './porto_OATD_pretain1.pth'.format(
            #     dataset, novel_map[novel_type], best_result["known_acc"], best_result["novel_acc"], best_result[
            #         "all_acc"])
            # torch.save(model.state_dict(), save_model_filename)


        print("Best Result: ", end='')
        for key, value in best_result.items():
            print('{}: {:.1f}'.format(key, value), end=' | ')
        print()

def Valid_binary_val(model, data, valid_batch_size, device, num_labels):
    data.batch_size = valid_batch_size
    y_pred = []
    binary = []  ## 0/1 result
    binary_pred = []  ## 0/1 probs
    target = []

    with torch.no_grad():
        for batch_encode_input, _, _, batch_seq_length, labels in data.iterate_val_data():
            batch_encode_input = torch.LongTensor(batch_encode_input).to(device)
            batch_seq_length = torch.LongTensor(batch_seq_length).to(device)
            labels = torch.LongTensor(labels).to(device)
            feats, outputs = model(batch_encode_input, batch_seq_length)
            probs = F.softmax(outputs, dim=-1)
            values, indices = torch.max(probs, -1)
            for i, true_label in enumerate(labels):
                if true_label < num_labels:
                    binary.append(1)
                    # binary_pred.append(1-probs[i][-1].item())
                    binary_pred.append(torch.max(probs[i][true_label]).item())
                else:
                    binary.append(0)
                    # binary_pred.append(1 - probs[i][-1].item())
                    binary_pred.append(torch.max(probs[i][:-1]).item())
                y_pred.append(indices[i].item())
                target.append(labels[i].item())

    # element_counts = Counter(y_pred)
    # print(element_counts)
    y_pred = np.array(y_pred)
    binary = np.array(binary)
    # element_counts = Counter(binary)
    # print(element_counts)
    binary_pred = np.array(binary_pred)
    target = np.array(target)

    print('F1-score:{}'.format(F1(target, y_pred, average='macro')))
    print('AUROC:{}'.format(auc(binary, binary_pred, average='macro')))
    precision, recall, _thresholds = metrics.precision_recall_curve(binary, binary_pred)
    area = metrics.auc(recall, precision)
    print('PR-AUC:{}'.format(area))

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    mapping = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size

@torch.no_grad()
def hungarian_evaluate(targets, predictions, offset=0):
    # Hungarian matching
    targets = targets - offset
    predictions = predictions - offset
    predictions_np = predictions.numpy()
    num_elems = targets.size(0)


    valid_idx = np.where(predictions_np >= 0)[0]
    predictions_sel = predictions[valid_idx]
    targets_sel = targets[valid_idx]
    num_classes = torch.unique(targets).numel()
    num_classes_pred = torch.unique(predictions_sel).numel()

    match = _hungarian_match(predictions_sel, targets_sel, preds_k=num_classes_pred,
                             targets_k=num_classes)  # match is data dependent
    reordered_preds = torch.zeros(predictions_sel.size(0), dtype=predictions_sel.dtype)
    for pred_i, target_i in match:
        reordered_preds[predictions_sel == int(pred_i)] = int(target_i)

    # Gather performance metrics
    reordered_preds = reordered_preds.numpy()
    acc = int((reordered_preds == targets_sel.numpy()).sum()) / float(
        num_elems)  # accuracy is normalized with the total number of samples not only the valid ones
    nmi = metrics.normalized_mutual_info_score(targets.numpy(), predictions.numpy())
    ari = metrics.adjusted_rand_score(targets.numpy(), predictions.numpy())

    return {'acc': acc * 100, 'ari': ari, 'nmi': nmi, 'hungarian_match': match}

@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):

            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes


    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def accuracy(logits, labels):

    preds = F.softmax(logits, -1)
    acc = 1.0 * (preds.argmax(-1) == labels)
    return acc.mean(0)
