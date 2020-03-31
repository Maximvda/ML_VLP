import os
import matplotlib.pyplot as plt
import csv
import numpy as np

def main():
    root = os.path.join(os.getcwd(),'AndroidLogs')
    result = os.path.join(os.getcwd(),'result')
    if not os.path.exists(result):
        os.mkdir(result)
    files = os.listdir(root)
    users = 0; id_list = []
    azimut = [];    pitch = []; roll = []
    for file in files:
        id = file.split("_")[0]
        if not (id in id_list):
            users += 1
            id_list.append(id)
        with open(os.path.join(root,file), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                azimut.append(float(row[0])*180/np.pi)
                pitch.append(float(row[1])*180/np.pi)
                roll.append(float(row[2])*180/np.pi)

    print(users)
    plt.figure()
    plt.hist(azimut, density=True, bins=100, label="Azimut")
    plt.title("Distribution of Azimut angle.")
    plt.xticks([-180,-90,0,90,180], ("", 'West', 'North', 'East', 'South'))
    plt.savefig(os.path.join(result,'azimut.png'))

    plt.figure()
    plt.hist(pitch, density=True, bins=100, label="Pitch")
    plt.title("Distribution of Pitch angle.")
    plt.xticks([-180,-90,0,90,180], ('-180°','-90°', '0°', '90°', '180°'))
    plt.savefig(os.path.join(result,'pitch.png'))

    plt.figure()
    plt.hist(pitch, density=True, bins=100, label="Roll")
    plt.title("Distribution of Roll angle.")
    plt.xticks([-90,-45,0,45,90], ('-90°','-45°', '0°', '45°', '90°'))
    plt.savefig(os.path.join(result,'roll.png'))

if __name__ == '__main__':
    main()
