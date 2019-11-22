import torch
import os

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
    dist = torch.sqrt((x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2)
    #print(dist*300)
    return torch.mean(dist).item()


#Makes a plot data and saves it to the result directory with given filename
#The title and labels of the plot are also taken as arguments
#If there are data_labels that means data is a list of lists each list gets plotted
def makePlot(data, filename, title, labels, result_root, data_labels=None):
    plt.figure(figsize=(10,5))
    plt.title(title)
    if data_labels == None:
        plt.plot(data)
    else:
        for i in range(0,len(data)):
            plt.plot(data[i], label=data_labels[i])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    resultpath = os.path.join(result_root, filename)
    plt.savefig(resultpath)

#Save all the relevant arguments in a textfile such that the model,
#the kind of data and test can be identified after training
def saveArguments(args):
    fileText = 'Number of features: {}\nBatch size: {}\nLearning rate: {}'.format(
                    args.nf, args.batch_size, args.learning_rate)
    fileName = os.path.join(args.result_root,'parameters.txt')
    f = open(fileName,'w')
    f.write(fileText)
    f.close()
