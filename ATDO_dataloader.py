import enum
import torch

import numpy as np
import pickle
from utils import get_dataset_args



class DataGenerator:
    def __init__(self, map_size=(51, 158), batch_size=256, sos=8059, eos=8060, dataset='porto'):
        print("Loading data...")
        self.map_size = map_size
        self.batch_size = batch_size
        self.sos = sos
        self.eos = eos
        self.dataset = dataset
        self.train_trajectories, self.train_traj_num = self.build_dataset('train')
        self.val_trajectories, self.val_traj_num = self.build_dataset('val')

    def build_dataset(self, data_type):
        data_name = "./data/{}/processed_{}_{}.csv".format(self.dataset, self.dataset, data_type)


        trajectories = sorted([
            eval(eachline) for eachline in open(data_name, 'r').readlines()
        ], key=lambda k: len(k))
        traj_num = len(trajectories)
        print("{} {} trajectories loading complete.".format(traj_num, data_type))

        return trajectories, traj_num

    def batch_pad(self, batch_x):
        max_len = max(len(x) for x in batch_x)
        batch_encode_input = [x + [0] * (max_len - len(x)) for x in batch_x]
        batch_decode_input = [[self.sos] + x + [0] * (max_len - len(x)) for x in batch_x]
        batch_decode_output = [x + [self.eos] + [0] * (max_len - len(x)) for x in batch_x]
        return batch_encode_input, batch_decode_input, batch_decode_output

    def inject_outliers(self, ratio=0.25, level=5, point_prob=0.3, vary=False, data_type=None):
        if data_type==None:
            raise ValueError('inject_outliers function args data_type is None')
        out_filename = '{}_outliers_{}.pkl'.format(self.dataset, data_type)
        if data_type=='train':
            traj_num = self.train_traj_num
            trajectories = self.train_trajectories
        else:
            traj_num = self.val_traj_num
            trajectories = self.val_trajectories

        size = int(traj_num * ratio)

        self.outlier_idx = selected_idx = np.random.choice(traj_num, size=size * 3, replace=False)
        self.detour_idx = detour_idx = selected_idx[:size]
        self.switching_idx = switching_idx = selected_idx[size:size * 2]
        self.gps_idx = gps_idx = selected_idx[size * 2:]

        detour_outliers = self.perturb_batch([trajectories[idx] for idx in detour_idx],
                                             level=np.random.randint(1, level), prob=point_prob)
        switching_outliers = self.shift_batch([trajectories[idx] for idx in switching_idx],
                                              level=level, prob=point_prob, vary=vary)
        gps_outliers = self.gps_batch([trajectories[idx] for idx in gps_idx],
                                      prob=point_prob)

        with open('./data/{}/NCD3/detour_'.format(self.dataset) + out_filename, 'wb') as fp:
            pickle.dump(dict(zip(detour_idx, detour_outliers)), fp)

        with open('./data/{}/NCD3/switching_'.format(self.dataset) + out_filename, 'wb') as fp:
            pickle.dump(dict(zip(switching_idx, switching_outliers)), fp)

        with open('./data/{}/NCD3/gps_'.format(self.dataset) + out_filename, 'wb') as fp:
            pickle.dump(dict(zip(gps_idx, gps_outliers)), fp)

        print("{} {} detour outliers injection is completed.".format(len(detour_outliers), data_type))
        print("{} {} switching outliers injection is completed.".format(len(switching_outliers), data_type))
        print("{} {} gps anomaly outliers injection is completed.".format(len(gps_outliers), data_type))

    def balance_data(self, train_unlabeled_idxs, train_labeled_idxs):
        if len(train_unlabeled_idxs) > len(train_labeled_idxs):
            exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
            train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

            if len(train_labeled_idxs) < len(train_unlabeled_idxs):
                diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
                train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
            else:
                assert len(train_labeled_idxs) == len(train_unlabeled_idxs)
            return train_unlabeled_idxs, train_labeled_idxs

        elif len(train_unlabeled_idxs) < len(train_labeled_idxs):
            exapand_unlabeled = len(train_labeled_idxs) //len(train_unlabeled_idxs)
            train_unlabeled_idxs = np.hstack([train_unlabeled_idxs for _ in range(exapand_unlabeled)])

            if len(train_labeled_idxs) > len(train_unlabeled_idxs):
                diff = len(train_labeled_idxs) - len(train_unlabeled_idxs)
                train_unlabeled_idxs = np.hstack((train_unlabeled_idxs, np.random.choice(train_unlabeled_idxs, diff)))
            else:
                assert len(train_labeled_idxs) == len(train_unlabeled_idxs)
            return train_unlabeled_idxs, train_labeled_idxs

        else:
            assert len(train_labeled_idxs) == len(train_unlabeled_idxs)
            return train_unlabeled_idxs, train_labeled_idxs

    def load_outliers(self, data_type=None, resample=False, novel_type=1, eta=0.5):
        if data_type == None:
            raise ValueError("load_outliers function args data_type is None")
        filename = '{}_outliers_{}.pkl'.format(self.dataset, data_type)
        if data_type=='train':
            traj_num = self.train_traj_num
            trajectories = self.train_trajectories
        else:
            traj_num = self.val_traj_num
            trajectories = self.val_trajectories
        labels = [0 for i in range(traj_num)]
        labeled_classes = [0, 1]
        unlabeled_classes = [2, 3]

        if novel_type==1:

            with open('./data/{}/NCD3/detour_'.format(self.dataset) + filename, 'rb') as fp:
                detour_idx = pickle.load(fp)
            for idx, o in detour_idx.items():
                trajectories[idx] = o
                labels[idx] = 1

            with open('./data/{}/NCD3/switching_'.format(self.dataset) + filename, 'rb') as fp:
                switching_idx = pickle.load(fp)
            for idx, o in switching_idx.items():
                trajectories[idx] = o
                labels[idx] = 2

            with open('./data/{}/NCD3/gps_'.format(self.dataset) + filename, 'rb') as fp:
                gps_idx = pickle.load(fp)
            for idx, o in gps_idx.items():
                trajectories[idx] = o
                labels[idx] = 3
        elif novel_type==2:
            with open('./data/{}/NCD3/detour_'.format(self.dataset) + filename, 'rb') as fp:
                detour_idx = pickle.load(fp)
            for idx, o in detour_idx.items():
                trajectories[idx] = o
                labels[idx] = 3

            with open('./data/{}/NCD3/switching_'.format(self.dataset) + filename, 'rb') as fp:
                switching_idx = pickle.load(fp)
            for idx, o in switching_idx.items():
                trajectories[idx] = o
                labels[idx] = 1

            with open('./data/{}/NCD3/gps_'.format(self.dataset) + filename, 'rb') as fp:
                gps_idx = pickle.load(fp)
            for idx, o in gps_idx.items():
                trajectories[idx] = o
                labels[idx] = 2

        elif novel_type==3:
            with open('./data/{}/NCD3/detour_'.format(self.dataset) + filename, 'rb') as fp:
                detour_idx = pickle.load(fp)
            for idx, o in detour_idx.items():
                trajectories[idx] = o
                labels[idx] = 2

            with open('./data/{}/NCD3/switching_'.format(self.dataset) + filename, 'rb') as fp:
                switching_idx = pickle.load(fp)
            for idx, o in switching_idx.items():
                trajectories[idx] = o
                labels[idx] = 3

            with open('./data/{}/NCD3/gps_'.format(self.dataset) + filename, 'rb') as fp:
                gps_idx = pickle.load(fp)
            for idx, o in gps_idx.items():
                trajectories[idx] = o
                labels[idx] = 1

        labeled_indices = np.where(np.isin(np.array(labels), labeled_classes))[0]
        unlabeled_indices = np.where(np.isin(np.array(labels), unlabeled_classes))[0]

        if data_type == 'train':
            unlabeled_indices, labeled_indices = self.balance_data(unlabeled_indices, labeled_indices)
            if resample == True:
                self.aug_train_trajectories = self.aug_trajectories(self.train_trajectories[:], eta)
                with open('./data/{}/NCD3/aug_train_trajectories_{}.pkl'.format(self.dataset, eta), 'wb') as fp:
                # with open('./data/NCD/3labels_aug_train_trajectories', 'wb') as fp:
                    pickle.dump(self.aug_train_trajectories, fp)
                print("aug data finished resample")
            else:
                with open('./data/{}/NCD3/aug_train_trajectories_{}.pkl'.format(self.dataset, eta), 'rb') as fp:
                # with open('./data/NCD/3labels_aug_train_trajectories', 'rb') as fp:
                    self.aug_train_trajectories = pickle.load(fp)

            self.train_trajectories = trajectories
            self.train_labels = labels
            trajectories = np.array(trajectories)
            labels = np.array(labels)
            self.known_trajectories = trajectories[labeled_indices]
            self.unknown_trajectories = trajectories[unlabeled_indices]
            self.known_labels = labels[labeled_indices]
            self.unknown_labels = labels[unlabeled_indices]
            self.labeled_indices = labeled_indices
            self.unlabel_indices = unlabeled_indices
        else:
            self.val_trajectories = trajectories
            self.val_labels = labels
        print("{} {} normal trajectories loading complete.".
              format(len(labels) - len(detour_idx) - len(switching_idx) - len(gps_idx), data_type))
        print("{} {} detour trajectories loading complete.".format(len(detour_idx), data_type))
        print("{} {} switching trajectories loading complete.".format(len(switching_idx), data_type))
        print("{} {} gps anomaly trajectories loading complete.".format(len(gps_idx), data_type))


    # 将数据集设为一个个batch
    def iterate_labeled_data(self):
        known_trajectories = self.known_trajectories

        known_labels = self.known_labels
        num = len(known_trajectories) // self.batch_size * self.batch_size
        known_trajectories = known_trajectories[:num]
        known_labels = known_labels[:num]
        for shortest_idx in range(0, num, self.batch_size):
            longest_idx = shortest_idx + self.batch_size
            known_trajectorie = known_trajectories[shortest_idx:longest_idx]
            known_label = known_labels[shortest_idx:longest_idx]
            known_length = [len(traj) for traj in known_trajectorie]
            known_trajectorie, _, _ = self.batch_pad(known_trajectorie)
            yield torch.LongTensor(known_trajectorie), torch.LongTensor(known_label), torch.LongTensor(known_length)
    def iterate_train_data(self, resample=False):

        # balance the labeled and unlabeled data
        known_trajectories = self.known_trajectories
        unknown_trajectories = self.unknown_trajectories
        known_labels = self.known_labels
        unknow_labels = self.unknown_labels
        unkonw_aug_trajectories = np.array(self.aug_train_trajectories, dtype=object)
        unkonw_aug_trajectories = unkonw_aug_trajectories[self.unlabel_indices]
        traj_num = len(known_trajectories)
        pad = self.batch_size - traj_num % self.batch_size
        known_trajectories = known_trajectories.tolist()
        unknown_trajectories = unknown_trajectories.tolist()
        unkonw_aug_trajectories = unkonw_aug_trajectories.tolist()
        known_labels = known_labels.tolist()
        unknow_labels = unknow_labels.tolist()
        known_trajectories = known_trajectories + known_trajectories[:pad]
        unknown_trajectories = unknown_trajectories + unknown_trajectories[:pad]
        unkonw_aug_trajectories = unkonw_aug_trajectories + unkonw_aug_trajectories[:pad]
        known_labels = known_labels + known_labels[:pad]
        unknow_labels = unknow_labels + unknow_labels[:pad]
        # know_labels = self.known_labels + self.known_labels[:pad]
        for shortest_idx in range(0, traj_num, self.batch_size):
            longest_idx = shortest_idx + self.batch_size
            know_label = known_labels[shortest_idx:longest_idx]
            unknow_label = unknow_labels[shortest_idx:longest_idx]
            batch_know_trajectories = known_trajectories[shortest_idx: longest_idx]
            batch_unknown_trajectories = unknown_trajectories[shortest_idx: longest_idx]
            batch_unkonw_aug_trajectories = unkonw_aug_trajectories[shortest_idx: longest_idx]
            batch_traj = batch_know_trajectories + batch_unknown_trajectories + batch_unkonw_aug_trajectories
            batch_seq_length = [len(traj) for traj in batch_traj]
            batch_encode_input, batch_decode_input, batch_decode_output = self.batch_pad(batch_traj)
            # batch_aug_encode_input, batch_aug_decode_input, batch_aug_decode_output = self.batch_pad(batch_aug_trajectories)
            # yield torch.LongTensor([batch_encode_input, batch_aug_encode_input]), torch.LongTensor([batch_decode_input, batch_aug_decode_input]), torch.LongTensor(
            #     [batch_decode_output, batch_aug_decode_output]), torch.LongTensor([batch_seq_length, batch_aug_seq_length]), torch.LongTensor(label)
            yield torch.LongTensor(batch_encode_input), torch.LongTensor(batch_decode_input), torch.LongTensor(
                batch_decode_output), torch.LongTensor(batch_seq_length), torch.LongTensor(know_label), torch.LongTensor(unknow_label)

    def iterate_val_data(self):
        traj_num = self.val_traj_num
        pad = self.batch_size - traj_num % self.batch_size
        trajectories = self.val_trajectories + self.val_trajectories[:pad]
        labels = self.val_labels + self.val_labels[:pad]
        for shortest_idx in range(0, traj_num, self.batch_size):
            longest_idx = shortest_idx + self.batch_size
            label = labels[shortest_idx:longest_idx]
            batch_trajectories = trajectories[shortest_idx: longest_idx]
            batch_seq_length = [len(traj) for traj in batch_trajectories]
            batch_encode_input, batch_decode_input, batch_decode_output = self.batch_pad(batch_trajectories)
            yield batch_encode_input, batch_decode_input, batch_decode_output, batch_seq_length, label

    def iterate_show_val_data(self):
        traj_num = self.val_traj_num
        pad = self.batch_size - traj_num % self.batch_size
        trajectories = self.val_trajectories
        labels = self.val_labels
        for shortest_idx in range(0, traj_num, self.batch_size):
            longest_idx = shortest_idx + self.batch_size
            label = labels[shortest_idx:longest_idx]
            batch_trajectories = trajectories[shortest_idx: longest_idx]
            batch_seq_length = [len(traj) for traj in batch_trajectories]
            batch_encode_input, batch_decode_input, batch_decode_output = self.batch_pad(batch_trajectories)
            yield batch_encode_input, batch_decode_input, batch_decode_output, batch_seq_length, label

    def _perturb_point(self, point, level, offset=None):
        # level = np.random.randint(1, level)
        map_size = self.map_size
        x, y = int(point // map_size[1]), int(point % map_size[1])
        if offset is None:
            offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
            x_offset, y_offset = offset[np.random.randint(0, len(offset))]
        else:
            x_offset, y_offset = offset
        if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
            x += x_offset * level
            y += y_offset * level
        return int(x * map_size[1] + y)

    # detour
    def perturb_batch(self, batch_x, level, prob):
        noisy_batch_x = []
        for traj in batch_x:
            noisy_batch_x.append([traj[0]] + [self._perturb_point(p, level)
                                              if not p == 0 and np.random.random() < prob else p
                                              for p in traj[1:-1]] + [traj[-1]])
        return noisy_batch_x

    # route-switching
    def shift_batch(self, batch_x, level, prob, vary=False):
        map_size = self.map_size
        noisy_batch_x = []
        if vary:
            level += np.random.randint(-2, 3)
            if np.random.random() > 0.5:
                prob += 0.2 * np.random.random()
            else:
                prob -= 0.2 * np.random.random()
        for traj in batch_x:
            anomaly_len = int((len(traj) - 2) * prob)
            anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            offset = [int(traj[anomaly_st_loc] // map_size[1]) - int(traj[anomaly_ed_loc] // map_size[1]),
                      int(traj[anomaly_st_loc] % map_size[1]) - int(traj[anomaly_ed_loc] % map_size[1])]
            if offset[0] == 0:
                div0 = 1
            else:
                div0 = abs(offset[0])  # abs返回绝对值
            if offset[1] == 0:
                div1 = 1
            else:
                div1 = abs(offset[1])

            if np.random.random() < 0.5:
                offset = [-offset[0] / div0, offset[1] / div1]
            else:
                offset = [offset[0] / div0, -offset[1] / div1]

            noisy_batch_x.append(traj[:anomaly_st_loc] +
                                 [self._perturb_point(p, level, offset) for p in traj[anomaly_st_loc:anomaly_ed_loc]] +
                                 traj[anomaly_ed_loc:])
        return noisy_batch_x

    def gps_batch(self, batch_x, prob):
        noisy_batch_x = []
        for traj in batch_x:
            anomaly_len = int(len(traj) * prob)
            anomaly_st_loc = np.random.randint(1, len(traj) - 1)
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            noisy_batch_x.append(traj[:anomaly_st_loc] +
                                 [np.random.randint(1, 8059) for p in traj[anomaly_st_loc:anomaly_ed_loc]] +
                                 traj[anomaly_ed_loc:])
        return noisy_batch_x

    def mask_and_remove_points(self, trajectory, mask_ratio=0.3):

        interval = int(1 / mask_ratio)  # Calculate the interval for sampling

        # Sample points at regular intervals
        sampled_trajectory = trajectory[::interval]

        return sampled_trajectory

    def aug_trajectories(self, trajectories, eta):
        return [self.mask_and_remove_points(traj, eta) for traj in trajectories]

    def traffic_batch(self, batch_x, prob):
        prob = 0.1
        noisy_batch_x = []
        for traj in batch_x:
            anomaly_len = 12
            anomaly_st_loc = np.random.randint(1, len(traj) - 1)
            for i in range(anomaly_len):
                if (np.random.random() > prob):
                    traj = traj[:anomaly_st_loc] + [traj[anomaly_st_loc]] + traj[anomaly_st_loc:]
                else:
                    traj = traj[:anomaly_st_loc + 1] + [traj[anomaly_st_loc + 1]] + traj[anomaly_st_loc + 1:]
            noisy_batch_x.append(traj)
        return noisy_batch_x

    def val_size(self, label):
        indices = [[] for i in range(4)]
        nums = [0 for i in range(4)]
        for i in range(self.batch_size):
            if (label[i] == 0):
                nums[0] += 1
                indices[0].append(i)
            elif (label[i] == 1):
                nums[1] += 1
                indices[1].append(i)
            elif (label[i] == 2):
                nums[2] += 1
                indices[2].append(i)
            else:
                nums[3] += 1
                indices[3].append(i)
        return nums, indices

    def get_trajectorys(self, data_type):
        if data_type != 'train' or data_type != 'val':
            raise ValueError('illegally data_type')
        if data_type == 'train':
            return self.train_trajectories
        else:
            return self.val_trajectories


if __name__ == '__main__':
    batch_size = 256
    dataset = 'chengdu'
    map_size, input_size, output_size, SOS_token, EOS_token = get_dataset_args(dataset)
    data = DataGenerator(map_size=map_size, batch_size=batch_size, sos=SOS_token, eos=EOS_token, dataset=dataset)
    data.inject_outliers(data_type='train')
    data.inject_outliers(data_type='val')
    data.load_outliers('train', resample=True)
    data.load_outliers('val')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_encode_input, batch_decode_input, batch_decode_output, batch_seq_length, labels, unlabels in data.iterate_train_data():
        x = batch_encode_input
        y = batch_seq_length
        print(x[256])
        print(x[256+256])
        print(x.shape)