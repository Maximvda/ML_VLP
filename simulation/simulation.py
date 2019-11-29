import numpy as np
import matplotlib.pyplot as plt

#Calculates the Received Signal Strength RSS
#d is the distance between TX and RX in mm
#phi is irradiation angle
#psi is incidence angle
def calculateRSS(d, psi):
    m = -np.log(2)/(np.log(np.cos(np.radians(15))))
    g_psi = 1
    phi = 0
    if 0 <= psi <= 90:
        #print(np.cos(np.radians(phi))**m*g_psi*np.cos(np.radians(psi)))
        H = ((m+1)*1.1)/(2*np.pi*d**2)*np.cos(np.radians(phi))**m*g_psi*np.cos(np.radians(psi))
        #print(H)
    else:
        H = 0
    return H


#Calculates the distance and incidence angle between RX and TX
def getDistAndAngle(pos_RX, pos_TX):
    x1 = pos_RX[0]; y1 = pos_RX[1]; z1 = pos_RX[2]
    x2 = pos_TX[0]; y2 = pos_TX[1]; z2 = pos_TX[2]
    print(y1-y2)

    d = ((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)**0.5
    l = ((x1-x2)**2+(y1-y2)**2)**0.5
    psi = 90 - np.degrees(np.arccos(l/d))

    return d, psi

#Simulate testbed and generate data
#First TX offset from origin: 50mm,50mm,1870mm
def testbed_simulation():
    RSS = []
    angle = []
    pos_RX = [0,0,0]
    pos_TX = [1500,50,1870]
    RX_x_sweep = np.arange(0,3000,10); pos_RX[1] = 50; pos_RX[2] = 0

    pos_RX = [-1,3,1]
    pos_TX = [3,3,-3]
    d, psi = getDistAndAngle(pos_RX, pos_TX)
    print(psi)

    for pos_RX[0] in RX_x_sweep:
        d, psi = getDistAndAngle(pos_RX, pos_TX)
        angle.append(psi)
        RSS.append(calculateRSS(d,psi))
        #pos_RX[0] = RX_x
        #print(pos_RX[0])

    #print(len(RSS))
    plt.plot(angle)
    plt.show()
