import torch
import os
import numpy as np
import sys

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

#Shows a grid of the possible positions of the measurement device
#The predicted positions for a batch are plotted in red
#The real position is plotted in green while a line shows the distance between predicted and real position
def visualise(target, prediction, pause=0.0001):
    plt.ion()
    plt.clf()
    plt.axis([0,1, 0, 1])
    target = target.cpu()
    prediction = prediction.cpu().detach()
    for i in range(0,len(target)):
        plt.plot([prediction[i,0], target[i,0]], [prediction[i,1], target[i,1]], 'k-')

    plt.plot(target[:,0], target[:,1], 'go')
    plt.plot(prediction[:,0], prediction[:,1], 'ro')
    plt.pause(pause)
    plt.show()

#Calculates the distance between two points a and b
#Euclidian distance = sqrt((ax-bx))^2+(ay-by)^2)
#The mean distance is calculated when x and y are lists of same length
def calcDistance(x,y):
    dist_2D = torch.sqrt((x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2)*300
    dist_2D = torch.mean(dist_2D).item()
    if len(x[0]) == 3:
        z_dist = torch.mean(torch.sqrt((x[:,2]-y[:,2])**2)*200).item()
        dist_3D = torch.sqrt(((x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2)*300**2+((x[:,2]-y[:,2])**2)*200**2)
        dist_3D = torch.mean(dist_3D).item()
        return {'2D': dist_2D, 'z': z_dist, '3D': dist_3D}
    else:
        return {'2D': dist_2D, '3D': np.inf}

#Calculates the bias or offset of the predicted points if there is any
#For example if predictions are always off by 1cm in x direction then bias -> x = 1
def calcBias(x,y):
    bias = [x[:,0]-y[:,0], x[:,1]-y[:,1]]
    return torch.mean(bias[0]).item(), torch.mean(bias[1]).item()


#Makes a plot data and saves it to the result directory with given filename
#The title and labels of the plot are also taken as arguments
#If there are data_labels that means data is a list of lists each list gets plotted
def makePlot(data, filename, title, labels, result_root, data_labels=None, colors=None, ticks=None):
    plt.figure(figsize=(10,5))
    plt.title(title)
    if data_labels == None:
        plt.plot(data)
        ind = data.index(min(data))
        plt.axvline(x=ind, color='red')
        plt.text(ind+0.1,data[1]/2,'Min = {} for TX = {}'.format(round(min(data),2),ind))
    else:
        for i in range(0,len(data)):
            if colors == None:
                if ticks is None:
                    plt.plot(data[i], label=data_labels[i])
                else:
                    plt.plot(ticks,data[i], label=data_labels[i])
            else:
                plt.plot(data[i], label=data_labels[i], color=colors[i])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    resultpath = os.path.join(result_root, filename)
    plt.savefig(resultpath)
    plt.close()

#Makes a heatmap plot for a given map
def makeHeatMap(map, filename, title, result_root):
    plt.imshow(map, cmap='viridis', vmin=0, vmax=25, interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x-axis (cm)')
    plt.ylabel('y-axis (cm)')
    plt.gca().invert_yaxis()
    resultpath = os.path.join(result_root, filename)
    plt.savefig(resultpath, bbox_inches='tight')
    plt.close()

#Make a print on the set line with certain offset
#This way multiple lines can track progress if multiple workers are used.
def printMultiLine(line, text, offset=0, end=False):
    line = line*3+offset+1
    for i in range(0,line):
        down()
    sys.stdout.write("\033[K")
    print('\r' + text, end="\r")
    if not end:
        for i in range(0,line):
            up()

def up():
    # My terminal breaks if we don't flush after the escape-code
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()

def down():
    # I could use '\x1b[1B' here, but newline is faster and easier
    sys.stdout.write('\n')
    sys.stdout.flush()

#prints a progress bar
def printProgBar(done, total_size):
    percent = ("{0:." + str(0) + "f}").format(100 * (done / float(total_size)))
    filledLength = int(50 * done // total_size)
    bar = "#" * filledLength + '-' * (50 - filledLength)
    string = '\r%s |%s| %s%% %s' % ("Progress", bar, percent, "")
    if done == total_size:
        string = ''
    return string
