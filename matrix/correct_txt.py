#读入txt，将第六列的所有数据改为1
import numpy as np
import os

def correct_txt(txt_path):
    # 读入txt文件
    txt = np.loadtxt(txt_path,delimiter=',')
    # 将第六列的所有数据改为1
    txt[:, 5] = 1
    # 保存为新的txt文件
    np.savetxt('new.txt', txt)
    print('new.txt文件已保存！')

if __name__ == '__main__':
    txt_path = '111.txt'
    correct_txt(txt_path)