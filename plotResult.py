import matplotlib.pyplot as plt
import re
import numpy as np

valsTest = []
valsTrain = []
with open('/home/santhosh/Projects/ANNCourse/ANN/Project/results/train3_1_5_1000/test.log') as f:
    lines = f.readlines()
    for i in range(1,len(lines)):
        valsTest.append(float(re.sub('[\n\t]', '', lines[i])))

    valsTest = np.asarray(valsTest)

with open('/home/santhosh/Projects/ANNCourse/ANN/Project/results/train3_1_5_1000/train.log') as f:
    lines = f.readlines()
    for i in range(1,len(lines)):
        valsTrain.append(float(re.sub('[\n\t]', '', lines[i])))

    valsTrain = np.asarray(valsTrain)

plt.plot(range(1, len(valsTrain)+1), valsTrain, 'r', label='Training accuracy')
plt.plot(range(1, len(valsTest)+1), valsTest, 'b', label='Testing accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.title('Training plot for CifarNet-Boosted')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
plt.legend(loc=0, borderaxespad=0.)
plt.show()
