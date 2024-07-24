from collections import defaultdict
import random
# sd_cnt = defaultdict(list)
# with open('./processed_porto_train.csv', mode='r') as f:
#     for eachline in f.readlines():
#         traj = eval(eachline)
#         s, d = traj[0], traj[-1]
#         sd_cnt[(s, d)].append(eachline)
# with open('./processed_porto_test.csv', mode='r') as f:
#     for eachline in f.readlines():
#         traj = eval(eachline)
#         s, d = traj[0], traj[-1]
#         sd_cnt[(s, d)].append(eachline)
# fout = open("./processed_porto_NCD.csv", mode='w')
# for trajs in sd_cnt.values():
#     for traj in trajs:
#         fout.write(traj)


#### split data
import pickle
with open('./processed_porto_NCD.csv', mode='r') as f:
    trajectories = [eval(eachline) for eachline in f.readlines()]
val_size = 0.3
val_count = int(len(trajectories) * val_size)
random.shuffle(trajectories)
val_data = trajectories[:val_count]
train_data = trajectories[val_count:]
with open('./NCD/porto_NCD_train.pkl', 'wb') as fp:
    pickle.dump(train_data, fp)

with open('./NCD/porto_NCD_val.pkl', 'wb') as fp:
    pickle.dump(val_data, fp)

# sd_cnt = defaultdict(list)
# for trajectory in train_data:
#     s, d = trajectory[0], trajectory[-1]
#     sd_cnt[(s, d)].append(trajectory)
# with open('./porto_NCD_train.csv', mode='a') as f:
#     for trajs in sd_cnt.values():
#         for traj in trajs:
#             f.write(traj)
#
# sd_cnt = defaultdict(list)
# for trajectory in val_data:
#     s, d = trajectory[0], trajectory[-1]
#     sd_cnt[(s, d)].append(trajectory)
# with open('./porto_NCD_val.csv', mode='a') as f:
#     for trajs in sd_cnt.values():
#         for traj in trajs:
#             f.write(traj)



