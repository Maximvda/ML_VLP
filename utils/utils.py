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
def calcDistance(x,y, area1=None, area2=None):
    z_dist = torch.mean(torch.sqrt((x[:,2]-y[:,2])**2)).item()
    if area1 is None:
        dist = torch.sqrt((x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2)
        return torch.mean(dist).item(), z_dist
    else:
        iList = checkArea(y, area1, area2)
        dist = []
        for i in iList:
            dist.append(torch.sqrt((x[i][0]-y[i][0])**2+(x[i][1]-y[i][1])**2).unsqueeze(0))
        if dist == []:
            return None
        else:
            dist = torch.cat(dist)
            return torch.mean(dist).item(), z_dist


#Calculates the distance between predicted and real position within a certain area
#The area is given through two points p1 and p2 defining a rectangle between them
def checkArea(positionList, area1, area2):
    iList = []

    p1 = [area1[0][0]/3000, area1[0][1]/3000]; p2 = [area1[1][0]/3000, area1[1][1]/3000]
    min_x1 = min((p1[0],p2[0])); max_x1 = max((p1[0],p2[0]))
    min_y1 = min((p1[1],p2[1])); max_y1 = max((p1[1],p2[1]))

    if area2 is not None:
        p1 = [area2[0][0]/3000, area2[0][1]/3000]; p2 = [area2[1][0]/3000, area2[1][1]/3000]
        min_x2 = min((p1[0],p2[0])); max_x2 = max((p1[0],p2[0]))
        min_y2 = min((p1[1],p2[1])); max_y2 = max((p1[1],p2[1]))

        for i in range(0,len(positionList)):
            y = positionList[i]
            if not (max_x1 < y[0].item() or y[0].item() < min_x1) or not (max_y1 < y[1].item() or y[1].item() < min_y1):
                continue
            if (max_x2 < y[0].item() < min_x2) or (max_y2 < y[1].item() < min_y2):
                continue
            iList.append(i)
    else:
        for i in range(0,len(positionList)):
            y = positionList[i]
            if (max_x1 < y[0].item() < min_x1) or (max_y1 < y[1].item() < min_y1):
                continue
            iList.append(i)

    return iList

def calcBias(x,y):
    bias = [x[:,0]-y[:,0], x[:,1]-y[:,1]]
    return torch.mean(bias[0]).item(), torch.mean(bias[1]).item()


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
    plt.close()

#Save all the relevant arguments in a textfile such that the model,
#the kind of data and test can be identified after training
def saveArguments(args):
    fileText = """Experiment: {}\n\nDATA PARAMETERS \n\n
                Simulation used: {}\n
                Transmitter configuartion: {}\n
                Number of transmitter inputs used: {}\n
                Dynamic data: {}\n\n\n
                MODEL PARAMETERS \n\n
                Model type: {}\n
                Extra layers: {}\n
                Number of features: {}\n\n\n
                LEARNING PARAMETERS\n\n
                Batch size: {}\n
                Learning rate: {}""".format(
                args.experiment,
                args.simulate,
                args.TX_config,
                args.TX_input,
                args.dynamic,
                args.model_type,
                args.extra_layers,
                args.nf,
                args.batch_size,
                args.learning_rate)
    fileName = os.path.join(args.result_root,'parameters.txt')
    f = open(fileName,'w')
    f.write(fileText)
    f.close()

def printProgBar(done, total_size):
    percent = ("{0:." + str(0) + "f}").format(100 * (done / float(total_size)))
    filledLength = int(50 * done // total_size)
    bar = "#" * filledLength + '-' * (50 - filledLength)
    print('\r%s |%s| %s%% %s' % ("Progress", bar, percent, ""), end = '\r')
