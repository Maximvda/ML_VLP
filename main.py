import os
import matplotlib.pyplot as plt
import csv
import numpy as np

def main():
    root = os.path.join(os.getcwd(),'AndroidLogs')
    result = os.path.join(os.getcwd(),'result')
    if not os.path.exists(result):
        os.mkdir(result)
    folders = os.listdir(root)

    azimut = [];    pitch = []; roll = []
    azimut_week = [];    pitch_week = []; roll_week = []
    for folder in folders:
        file_path = os.path.join(root,folder)
        files = os.listdir(file_path)
        users = 0; id_list = []
        for file in files:
            id = file.split("_")[0]
            if not (id in id_list):
                users += 1
                id_list.append(id)
            with open(os.path.join(file_path,file), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    azimut.append(float(row[0])*180/np.pi)
                    pitch.append(float(row[1])*180/np.pi)
                    roll.append(float(row[2])*180/np.pi)
                    if folder in ["day1", "day2", "day3", "day4", "day5", "day6", "day7"]:
                        azimut_week.append(float(row[0])*180/np.pi)
                        pitch_week.append(float(row[1])*180/np.pi)
                        roll_week.append(float(row[2])*180/np.pi)

        print("Number of users for {} is: {}".format(folder,users))

    plt.figure()
    plt.hist(azimut, density=True, bins=100, label="Azimut")
    plt.title("Distribution of Azimut angle.")
    plt.xticks([-180,-90,0,90,180], ("South", 'West', 'North', 'East', 'South'))
    plt.xlabel("Orientation")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(result,'azimut.png'))

    plt.figure()
    plt.hist(pitch, density=True, bins=100, label="Pitch")
    plt.title("Distribution of Pitch angle.")
    plt.xticks([-90,-45,0,45,90], ('-90°','-45°', '0°', '45°', '90°'))
    plt.xlabel("Angle")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(result,'pitch.png'))

    plt.figure()
    plt.hist(roll, density=True, bins=100, label="Roll")
    plt.title("Distribution of Roll angle.")
    plt.xticks([-180,-90,0,90,180], ('-180°','-90°', '0°', '90°', '180°'))
    plt.xlabel("Angle")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(result,'roll.png'))

    plt.figure()
    plt.hist(azimut_week, density=True, bins=100, label="Azimut")
    plt.title("Distribution of Azimut angle.")
    plt.xticks([-180,-90,0,90,180], ("South", 'West', 'North', 'East', 'South'))
    plt.xlabel("Orientation")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(result,'azimut_week.png'))

    plt.figure()
    plt.hist(pitch_week, density=True, bins=100, label="Pitch")
    plt.title("Distribution of Pitch angle.")
    plt.xticks([-90,-45,0,45,90], ('-90°','-45°', '0°', '45°', '90°'))
    plt.xlabel("Angle")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(result,'pitch_week.png'))

    plt.figure()
    plt.hist(roll_week, density=True, bins=100, label="Roll")
    plt.title("Distribution of Roll angle.")
    plt.xticks([-180,-90,0,90,180], ('-180°','-90°', '0°', '90°', '180°'))
    plt.xlabel("Angle")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(result,'roll_week.png'))

if __name__ == '__main__':
    main()
