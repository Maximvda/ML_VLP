import numpy as np
import os
import pickle

from utils.utils import printProgBar

#Simulate testbed and generate data
def testbed_simulation(dataroot, verbose):
    file = os.path.join(dataroot,'simulationdata.data')
    if not os.path.exists(file):
        print("Running simulation and storing data")
        channel_data = np.zeros((36,3,300,300,2))
        #Get the TX and RX positions for which measurements want to be generated
        pos_TX_192 = getPositionTX()
        pos_TX_176 = [[pos[0], pos[1], pos[2]-160] for pos in pos_TX_192]
        pos_TX = [pos_TX_192, pos_TX_176]
        pos_RX = [[x, y, 0] for x in np.arange(0,3000,10) for y in np.arange(0,3000,10)]

        counter = 0
        for RX in pos_RX:
            counter += 1
            if verbose:
                print(printProgBar(counter, len(pos_RX)), end='\r')
            RSS = []; RSS1 = []; RSS2 = []
            for i in range(len(pos_TX)):
                for TX in pos_TX[i]:
                    d, psi = getDistAndAngle(RX, TX)
                    #Generate three measurements with different noise
                    RSS.append(calculateRSS(d,psi))
                    RSS1.append(calculateRSS(d,psi))
                    RSS2.append(calculateRSS(d,psi))


                channel_data[:,0, int(RX[0]/10), int(RX[1]/10), i] = RSS
                channel_data[:,1, int(RX[0]/10), int(RX[1]/10), i] = RSS1
                channel_data[:,2, int(RX[0]/10), int(RX[1]/10), i] = RSS2

        dict = {'channel_data': channel_data,
                'pos_TX': pos_TX,
                'pos_RX': pos_RX}

        with open(file, 'wb') as f:
            pickle.dump(dict, f)


#First TX offset from origin:X = 230mm and y =170mm
def getPositionTX():
    ##Actual position of the TX (roughly measured)
    return [[230, 170, 1920], [730, 175, 1920], [1240, 170, 1920], [1740, 170, 1920], [2240, 170, 1920], [2740, 170, 1920],
            [230, 670, 1920], [720, 725, 1835], [1230, 670, 1920], [1735, 670, 1920], [2225, 725, 1835], [2725, 670, 1920],
            [230, 1170, 1920], [730, 1170, 1920], [1240, 1170, 1920], [1745, 1170, 1920], [2245, 1170, 1920], [2735, 1170, 1920],
            [230, 1670, 1920], [730, 1670, 1920], [1240, 1670, 1920], [1745, 1670, 1920], [2245, 1670, 1920], [2735, 1670, 1920],
            [230, 2170, 1920], [720, 2225, 1835], [1235, 2170, 1920], [1720, 2170, 1920], [2220, 2225, 1835], [2710, 2170, 1920],
            [215, 2670, 1920], [715, 2670, 1920], [1245, 2670, 1920], [1730, 2670, 1920], [2245, 2670, 1920], [2730, 2670, 1920]]

#Calculates the Received Signal Strength RSS
#d is the distance between TX and RX in mm
#phi is irradiation angle
#psi is incidence angle
def calculateRSS(d, psi):
    #Lambertian order
    m = -np.log(2)/(np.log(np.cos(np.radians(15))))
    g_psi = 1
    phi = psi
    #Line of sight path loss
    if 0 <= psi <= 90:
        H = ((m+1)*1.1)/(2*np.pi*d**2)*np.cos(np.radians(phi))**m*g_psi*np.cos(np.radians(psi))
    else:
        H = 0
    #Adding white gaussian noise of 2%
    noise = np.random.normal(0,H*0.05,1)
    return H+noise


#Calculates the distance and incidence angle between RX and TX
def getDistAndAngle(pos_RX, pos_TX):
    x1 = pos_RX[0]; y1 = pos_RX[1]; z1 = pos_RX[2]
    x2 = pos_TX[0]; y2 = pos_TX[1]; z2 = pos_TX[2]

    d = ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
    l = ((x1-x2)**2+(y1-y2)**2)**0.5
    psi =  90 - np.degrees(np.arccos(l/d))

    return d, psi
