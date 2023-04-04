import numpy as np
import os
import matplotlib.pyplot as plt

local_map_path = "./Local_Mapping/time_train.txt"
global_map_path = "./time_train.txt"



with open(local_map_path, "r") as l:
    lt = np.array(l.read().split('\n'))[:100].astype(np.float32) * 1000

with open(global_map_path, "r") as l:
    gt = np.array(l.read().split('\n'))[:100].astype(np.float32) * 1000




# local_time = np.array([open(local_map_path, "r").read()])
# global_time = np.array([open(global_map_path, "r").read()])

# print(local_time.shape)
t = np.arange(lt.shape[0])
print(t)

plt.plot(t, gt, t, lt)
plt.xlabel('Iterations')
plt.ylabel('Time (ms)')
plt.legend(['Global Mapping', 'Local Mapping'])

plt.show()


# local_map_path.close()