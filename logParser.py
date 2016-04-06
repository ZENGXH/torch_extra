import re
import scipy
import pylab
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import glob
from sparnn.helpers import visualization
# print "argument2 is " + sys.argv[2]
# fil = open('hko-record/HKO-prediction-'+ sys.argv[1] + '.log','r')
# fil
# print fil

def plotLoss(path, nameOfFile, average_interval):
    
    #fileList = glob.glob(path + "*out*")
    #for fil in fileList:
        # print fil
        # tx = fil.read()
    #    visualization.visualize_loss(fil)

    fileList = glob.glob(path + "*" +nameOfFile+ "*" )
    print(nameOfFile)
    print(fileList)
    for target_file in fileList:
    #    print(target_file)
        loss_file = open(target_file, 'r')
        lines = loss_file.readlines()
        loss_file.close()

        iter_list = []
        loss_list = []
        iter_num = 0

        flag = 1
        accum_loss = 0
        accum_loss_list = []
        accum_loss_iter_list = []
        #average_interval = 8000

        print('average_interval of plotting:', average_interval)

        lossFigure = plt.figure(1)
        trainFigure = plt.figure(2)

        plt.figure(1)
        for ind, line in enumerate(lines):
            if "valid" in line:
                continue
            if "err" not in line:
                continue
            loss_num = line.split('err:\t')[1]
            iter_num += 1
            loss_list.append(float(loss_num))
            iter_list.append(int(iter_num))
            accum_loss += float(loss_num)

            if 0 == (int(iter_num) % average_interval):
                accum_loss_iter_list.append(int(iter_num))
                accum_loss_list.append(float(accum_loss))
                accum_loss = 0
        print('total training iter: ', iter_num)
        plt.xlabel('iteration number')
        plt.ylabel('training loss')
        plt.plot(accum_loss_iter_list[:], accum_loss_list[:])
        #print(accum_loss_list)
        title = nameOfFile + 'trainingLoss_ave'+ str(average_interval)  + '.png'
        plt.title(title)
        plt.grid()
        plt.savefig(path + title)
        #plt.show()

        # plot the validation loss for each epoch
        plt.figure(2)

        iter_list = []
        loss_list = []
        iter_num = 0

        flag = 1
        accum_loss = 0
        accum_loss_list = []
        accum_loss_iter_list = []

        loss_file = open(target_file, 'r')
        lines = loss_file.readlines()
        loss_file.close()

        for ind, line in enumerate(lines):
            #print('.')
            #if "validation" not  in line:
            if "valid" and "score" not in line:
                #print(line)
                continue
            #print(line)
            #loss_num = line.split('is \t')[1]
            loss_num = line.split('score:\t')[1]
            #print(loss_num)
            iter_num += 1
            loss_list.append(float(loss_num))

            iter_list.append(int(iter_num))
            accum_loss += float(loss_num)

            if 0 == (int(iter_num) % 10):
                accum_loss_iter_list.append(int(iter_num))
                accum_loss_list.append(float(accum_loss))
                accum_loss = 0
        print('total valid iter: ', iter_num)
        plt.xlabel('epoch number')
        plt.ylabel('validation loss')
        #plt.plot(loss_list[:], iter_list[:])

        plt.plot(accum_loss_iter_list[:], accum_loss_list[:])
        title = nameOfFile + 'validLoss.png'
        plt.title(title)
        plt.grid()
        plt.savefig(path + title)
        plt.show()
        return

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("require path nameOfFile")
        sys.exit(1)
    elif len(sys.argv) == 2: # only input name
        nameOfFile = sys.argv[1]
        average_interval = 8000
    else: # also input interval

        nameOfFile = sys.argv[1]
        average_interval = sys.argv[2]
    path = '../of_record/'
    print(path + nameOfFile + str(average_interval))
    plotLoss(path, nameOfFile, int(average_interval))


