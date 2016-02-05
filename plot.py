import matplotlib.pyplot as plt
import re
import numpy as np

valsTest = []
valsTrain = []
with open('results/cifar_boosted/test.log') as f:
    lines = f.readlines()
    for i in range(1,len(lines)):
        valsTest.append(float(re.sub('[\n\t]', '', lines[i])))

    valsTest = np.asarray(valsTest)

with open('results/cifar_boosted/train.log') as f:
    lines = f.readlines()
    for i in range(1,len(lines)):
        valsTrain.append(float(re.sub('[\n\t]', '', lines[i])))

    valsTrain = np.asarray(valsTrain)

plt.plot(valsTrain, 'r')
plt.plot(valsTest, 'b')
plt.show()
