import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ATDO_dataloader import DataGenerator
from utils import symmetric_mse_loss, entropy, get_dataset_args
from model import build_NCD, Encoder, Classifier, NCD, sim_matrix, SimNet
from eval import Valid_train, Valid_val

# args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1:val=GPS  2:val=detour  3:val=switching
novel_type = 1
novel_map = {1: 'switching and navigation', 2: "detour and navigation", 3: "detour and switching"}
dataset = 'porto'
# dataset = 'chengdu'
map_size, input_size, output_size, SOS_token, EOS_token = get_dataset_args(dataset)

num_labels = 2
num_unlabels = 2

raw_embedding_dim = 256
embedding_dim = 128
hidden_dim = 256
z_dim = 128
num_heads = 2

epochs = 100
layer = 1
batch_size = 256
valid_batch_size = 1024
lr = 0.001
lr2 = lr*5
eta = 0.5
gamma = 0.8

a1 = 0
a2 = 0
embedding1 = []
embedding2 = []
with open('./embedding/{}/embedding_osr1_SD.csv'.format(dataset, novel_type), 'r') as f:
    for line in f:
        if (a1 == 0):
            a1 += 1
        else:
            embedding1.append(eval(line))
embedding1 = torch.FloatTensor(np.array(embedding1))
with open('./embedding/{}/embedding_osr1_region.csv'.format(dataset, novel_type), 'r') as f:
    for line in f:
        if (a2 == 0):
            a2 += 1
        else:
            embedding2.append(eval(line))
embedding2 = torch.FloatTensor(np.array(embedding2))
embedding = torch.cat((embedding1, embedding2), dim=1).to(device)


def Train(model, data, optimizer, simnet, optimizer_simnet):
    data.batch_size = batch_size
    model.train()
    total_loss = 0
    for batch_encode_input, _, _, batch_seq_length, labels, unknow_labels in data.iterate_train_data():
        # print(batch_encode_input[19:43])
        # print(labels[19:43])
        # print(batch_encode_input[528:534])
        # print(unknow_labels[16:23])
        # print(batch_encode_input[536:542])
        # print(unknow_labels[24:30])
        # np_batch_encode_input = np.array(batch_encode_input)
        # np_labels = np.array(labels)
        # np_unknown_labels = np.array(unknow_labels)
        batch_encode_input, batch_seq_length, labels = batch_encode_input.to(device), batch_seq_length.to(device), labels.to(device)
        batch_l = labels.shape[0]
        model.train()
        model_ = build_NCD(raw_embedding_dim, embedding_dim, hidden_dim, z_dim, num_labels, num_unlabels, embedding, device)
        model_ = model_.cuda()
        model_.load_state_dict(model.state_dict())
        feat, logits = model_(batch_encode_input, batch_seq_length)
        logits_l = logits[:batch_l]
        logits_u_w, logits_u_s = logits[batch_l:2*batch_l], logits[2*batch_l:]
        feat_l = feat[:batch_l]
        feat_u_w = feat[batch_l:2*batch_l]
        feats = torch.cat((feat_l, feat_u_w), 0)
        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs_pl, targets_u_pl = torch.max(pseudo_label, dim=-1)
        mask_pl = max_probs_pl.ge(gamma).float()

        feats = feats.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        feats_t = torch.transpose(feats, 0, 1)
        feat_pairs = torch.cat((feats, feats_t), 2).view(-1, feats.shape[2] * 2)
        sim_feat = simnet(feat_pairs).view(-1, feats.shape[0])

        class_logit = torch.cat((logits_l, logits_u_w), 0)
        sim_prob = sim_matrix(F.softmax(class_logit, dim=1), F.softmax(class_logit, dim=1))

        loss_pair = symmetric_mse_loss(sim_prob.view(-1), sim_feat.view(-1)) / sim_feat.view(-1).shape[0]
        loss_reg = entropy(torch.mean(F.softmax(class_logit, dim=1), 0), input_as_probabilities=True)
        loss_ce_supervised = F.cross_entropy(class_logit[:batch_l], labels)
        loss_ce_pseudo = (F.cross_entropy(logits_u_s, targets_u_pl, reduction='mean') * mask_pl).mean()
        loss_ce = loss_ce_supervised + loss_ce_pseudo

        loss = loss_pair - loss_reg + loss_ce
        model_.zero_grad()
        feat, logits = model_(batch_encode_input, batch_seq_length)
        logits_l = logits[:batch_l]
        loss_ce_supervised = F.cross_entropy(logits_l, labels)
        optimizer_simnet.zero_grad()
        loss_ce_supervised.backward()
        optimizer_simnet.step()

        feat, logits = model(batch_encode_input, batch_seq_length)
        logits_l = logits[:batch_l]
        logits_u_w, logits_u_s = logits[batch_l:2*batch_l], logits[2*batch_l:]

        feat_l = feat[:batch_l]
        feat_u_w = feat[batch_l:2 * batch_l]
        feats = torch.cat((feat_l, feat_u_w), 0)

        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs_pl, targets_u_pl = torch.max(pseudo_label, dim=-1)
        mask_pl = max_probs_pl.ge(gamma).float()

        feats = feats.unsqueeze(0).repeat(feats.shape[0], 1, 1)
        feats_t = torch.transpose(feats, 0, 1)
        feat_pairs = torch.cat((feats, feats_t), 2).view(-1, feats.shape[2] * 2)

        # no gradients for the simnet parameters
        with torch.no_grad():
            sim_feat = simnet(feat_pairs).view(-1, feats.shape[0])
        class_logit = torch.cat((logits_l, logits_u_w), 0)
        sim_prob = sim_matrix(F.softmax(class_logit, dim=1), F.softmax(class_logit, dim=1))

        loss_pair = symmetric_mse_loss(sim_prob.view(-1), sim_feat.view(-1)) / sim_feat.view(-1).shape[0]
        loss_reg = entropy(torch.mean(F.softmax(class_logit, dim=1), 0), input_as_probabilities=True)
        loss_ce_supervised = F.cross_entropy(class_logit[:batch_l], labels)
        loss_ce_pseudo = (F.cross_entropy(logits_u_s, targets_u_pl, reduction='none') * mask_pl).mean()
        loss_ce = loss_ce_supervised + loss_ce_pseudo

        final_loss = loss_pair - loss_reg + loss_ce

        total_loss += final_loss.item()
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

    print("total_loss:{:.4f}".format(total_loss))


if __name__ == "__main__":
    data = DataGenerator(map_size=map_size, batch_size=batch_size, sos=SOS_token, eos=EOS_token, dataset=dataset)
    data.load_outliers('train', resample=True, novel_type=novel_type, eta=eta)
    data.load_outliers('val', resample=True, novel_type=novel_type)

    encoder = Encoder(raw_embedding_dim, embedding_dim, hidden_dim, embedding, device)
    classifier = Classifier(embedding_dim, z_dim, num_labels, num_unlabels)
    model = NCD(encoder, classifier).to(device)
    simnet = SimNet(embedding_dim*2, 100, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_simnet = torch.optim.Adam(simnet.params(), lr=lr2)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer_simnet, step_size=30, gamma=0.8)
    print("dataset: [{}], novel class :[{}]".format(dataset, novel_map[novel_type]))
    print("embedding dim: {}, N dim: {}, eta: {}, gamma: {}".format(embedding_dim, hidden_dim, eta, gamma))
    # model.load_state_dict(torch.load('./porto_OATD_pretain1.pth'), strict=False)
    for epoch in range(epochs):
        print(f'Epoch {epoch}: ')
        Train(model, data, optimizer, simnet, optimizer_simnet)
        scheduler.step()
        scheduler2.step()
        print('Train dataset metrics:')
        Valid_train(model, data, valid_batch_size, device)
        print('Val dataset metrics:')
        Valid_val(model, data, valid_batch_size, device, num_labels, dataset, novel_map, novel_type)


