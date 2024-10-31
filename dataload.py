#%%
import numpy as np

print(np.abs(2))
#%%
dir = "E:\ChristianMusaeus\Data\Eyes_closed_marked"
from scipy.io import loadmat

data = loadmat(f"{dir}/10002.mat")
#%%
datanp = np.array([data])
#print(datanp)
#print(datanp.shape)

#for i in [10,11]:
   # for j in ["EC", "EO"]:
       # data = loadmat(f"{dir}/S{i}_restingPre_{j}.mat")
       # np.save(f"{dir}/S{i}_restingPre_{j}.npy", data)
       # print(f"{dir}/S{i}_restingPre_{j}.npy saved")
# %%
data.items()


#%%
# load .set file
import mne
dir = "E:/ChristianMusaeus/Data/Eyes_closed_marked"
data = mne.io.read_raw_eeglab(f"{dir}/10002_p01_epoched_EyesOpen_marked.set")
#%%
