import csv
import numpy as np
import pandas as pd
import os 

if not os.path.exists("./data"):
    os.makedirs("./data")
    os.makedirs("./data/train")
    os.makedirs("./data/test")
    
SMD_machine = ["machine-1-1","machine-1-2","machine-1-3","machine-1-4","machine-1-5","machine-1-6","machine-1-7","machine-1-8","machine-2-1","machine-2-2","machine-2-3","machine-2-4","machine-2-5","machine-2-6","machine-2-7","machine-2-8","machine-2-9","machine-3-1","machine-3-2","machine-3-3","machine-3-4","machine-3-5","machine-3-6","machine-3-7","machine-3-8","machine-3-9","machine-3-10","machine-3-11"]

for i in SMD_machine:
    traindata = np.genfromtxt("../datasets/ServerMachineDataset/train/%s.txt"%i, delimiter=",")
    np.save("./data/train/%s.npy"%i, traindata)
    testdata = np.genfromtxt("../datasets/ServerMachineDataset/test/%s.txt"%i, delimiter=",")
    np.save("./data/test/%s.npy"%i, testdata)
    
import csv
import numpy as np
import pandas as pd
import os

'''
put SWAT data in ../swat/train/machine-1-1.txt
put SWAT data in ../swat/test/machine-1-1.txt
put SWAT data in ../swat/test_label/machine-1-1.txt
''' 

df = pd.read_csv('../datasets/SWAT/SWaT_Dataset_Normal_v1.csv')
df = df.drop(columns=['Unnamed: 0','Unnamed: 52'])
traindata = df[1:].to_numpy(dtype=np.float64)[21600:]
print("SWAT shape is " , traindata.shape)

df = pd.read_csv('../datasets/SWAT/SWaT_Dataset_Attack_v0.csv')
y = df['Normal/Attack'].to_numpy()
df = df.drop(columns=[' Timestamp', 'Normal/Attack'])
testdata = df.to_numpy(dtype=np.float64)
print("SWAT shape is " , testdata.shape)

test_label = []
for i in y:
    if i == 'Attack':
        test_label.append(1)
    else:
        test_label.append(0)
test_label = np.array(test_label)
print("label shape is " , test_label.shape)

np.save("./data/train/swat.npy", traindata)
np.save("./data/test/swat.npy", testdata)


# np.savetxt("../datasets/swat/test_label/machine-1-1.txt", test_label, delimiter='\n', fmt='%d')

'''
put SWAT data in ../wadi/train/machine-1-1.txt
put SWAT data in ../wadi/test/machine-1-1.txt
put SWAT data in ../wadi/test_label/machine-1-1.txt
''' 

a = str(open('../datasets/WADIA2/WADI_14days_new.csv', 'rb').read(), encoding='utf8').split('\n')[5: -1]
a = '\n'.join(a)
with open('train1.csv', 'wb') as f:
    f.write(a.encode('utf8'))
a = pd.read_csv('train1.csv', header=None)
a = a.to_numpy()[:, 3:]
nan_cols = []
for j in range(a.shape[1]):
    for i in range(a.shape[0]):
        if a[i][j] != a[i][j]:
            nan_cols.append(j)
            break
train = np.delete(a, nan_cols, axis=1)
traindata=train.astype(np.float64)[21600:]
print("WADI shape is " , traindata.shape)

df = pd.read_csv('../datasets/WADIA2/WADI_attackdataLABLE.csv')
test = df.to_numpy()[2:, 3:-1]
test = test.astype(np.float64)
testdata = np.delete(test, nan_cols, axis=1)
print("WADI test shape ", testdata.shape)

test_label = df.to_numpy()[2:, -1]
test_label = test_label.astype(np.int32)
print(test_label.shape)
for i in range(len(test_label)):
    if test_label[i] <= 0:
        test_label[i] = 1
    else:
        test_label[i] = 0
        
np.save("./data/train/wadi.npy", traindata)
np.save("./data/test/wadi.npy", testdata)
