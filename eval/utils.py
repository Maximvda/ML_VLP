import time

import matplotlib.pyplot as plt

#Shows a grid of the possible positions of the measurement device
#The predicted positions for a batch are plotted
#The target position is also shown with a arrow from the prediction to the target
def visualise(target, prediction):
    plt.ion()
    plt.clf()
    plt.axis([0,1, 0, 1])
    target = target.cpu()
    prediction = prediction.cpu().detach()
    for i in range(0,len(target)):
        plt.plot([prediction[i,0], target[i,0]], [prediction[i,1], target[i,1]], 'k-')

    plt.plot(target[:,0], target[:,1], 'go')
    plt.plot(prediction[:,0], prediction[:,1], 'ro')
    plt.pause(0.0101)
    plt.show()
