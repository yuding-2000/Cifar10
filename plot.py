import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib.ticker import FuncFormatter
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from train import target2class
with open('log_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

train_acc=[]
train_loss=[]
test_acc=[]
clas = np.zeros(10)

for i in range(len(lines)):
    line = lines[i]
    if i % 16 < 5:  # log of train
        tmp = line.split(':')
        train_loss.append(eval(tmp[-1][:8]))
        tmp = tmp[-2].split('%')
        train_acc.append(eval(tmp[0])/100)
    elif i % 16 == 5:  # log of test
        tmp = line.split('%')[0].split(':')
        test_acc.append(eval(tmp[-1])/100)
    else:  # log of class
        tmp = line.split('is ')[-1]
        clas[i%16-6]=eval(tmp[:-2])/100
def percent_formatter(x, pos):
    return f'{x * 100:.0f}%'  # 将小数转换为百分比

# 应用格式化器
# plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_formatter))


# plt.plot(range(1,len(train_acc)+1),train_acc,label='Train Accuracy')
# plt.plot(range(5,5*len(test_acc)+1,5),test_acc,label='Test Accuracy')
plt.plot(range(1,len(train_loss)+1),train_loss,label='Train Loss')
plt.grid()
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim(0.5,1)
# plt.yticks([i/100 for i in range(50,101,10)])
plt.show()


# plt.bar(x=[target2class(i)[:2] for i in range(10)], height=clas)
# plt.yticks([i/100 for i in range(75,101,5)])
# plt.xlabel('种类')
# plt.ylabel('识别准确率')
# plt.ylim(0.75,1)
# plt.grid()
# plt.show()

