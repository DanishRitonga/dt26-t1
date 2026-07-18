import numpy as np

data1 = np.load("./dataset-task1/train/train_speed_m1_1_11160.npy")
data2 = np.load("./dataset-task1/train/train_speed_m2_1_5039.npy")
mask = np.load("./dataset-task1/static/active_mask.npy")
test = np.load("./dataset-task1/test/test_X_hist.npy")

# print(data1)
# print(data1.shape)
#
# print(data2)
# print(data2.shape)

print(test)
print(test.shape)
