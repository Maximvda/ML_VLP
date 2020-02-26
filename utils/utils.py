import torch
import os

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

def convolution2d(meas_array, max=True):
    likelyCell = []
    for i in range(0,4):
        for j in range(0,4):
            likelyCell.append(meas_array[0+j+6*i]+meas_array[1+j+6*i]+meas_array[2+j+6*i]+
                meas_array[0+j+6*(i+1)]+meas_array[1+j+6*(i+1)]+meas_array[2+j+6*(i+1)]+
                meas_array[0+j+6*(i+2)]+meas_array[1+j+6*(i+2)]+meas_array[0+j+6*(i+2)])
    if max:
        ind = np.argmax(likelyCell)
    else:
        ind = np.argmin(likelyCell)
    return int(ind+7+6*np.floor(ind/4))

#Shows a grid of the possible positions of the measurement device
#The predicted positions for a batch are plotted in red
#The real position is plotted in green while a line shows the distance between predicted and real position
def visualise(target, prediction, pause=0.0001):
    plt.switch_backend('TkAgg')
    plt.ion()
    plt.clf()
    plt.axis([-1,1, -1, 1])
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
    return torch.mean(dist).item()


def calcBias(x,y):
    bias = [x[:,0]-y[:,0], x[:,1]-y[:,1]]
    return torch.mean(bias[0]).item(), torch.mean(bias[1]).item()


#Makes a plot data and saves it to the result directory with given filename
#The title and labels of the plot are also taken as arguments
#If there are data_labels that means data is a list of lists each list gets plotted
def makePlot(data, filename, title, labels, result_root, data_labels=None, colors=None):
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
                plt.plot(data[i], label=data_labels[i])
            else:
                plt.plot(data[i], label=data_labels[i], color=colors[i])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    resultpath = os.path.join(result_root, filename)
    plt.savefig(resultpath)
    plt.close()

def makeHeatMap(map, filename, title, result_root):
    #plt.figure(figsize=(10,10))
    plt.imshow(map, cmap='viridis', vmin=0, vmax=45, interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('x-axis (cm)')
    plt.ylabel('y-axis (cm)')
    plt.gca().invert_yaxis()
    resultpath = os.path.join(result_root, filename)
    plt.savefig(resultpath, bbox_inches='tight')
    plt.close()

#Save all the relevant arguments in a textfile such that the model,
#the kind of data and test can be identified after training
def saveArguments(args):
    fileText = """Experiment: {}\n\nDATA PARAMETERS \n\n
                MODEL PARAMETERS \n\n
                Model type: {}\n
                Extra layers: {}\n
                Number of features: {}\n\n\n
                LEARNING PARAMETERS\n\n
                Batch size: {}\n
                Learning rate: {}""".format(
                args.experiment,
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
