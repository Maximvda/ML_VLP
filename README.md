# Visible Light Positioning with Machine Learning

The goal of this project is to use machine learning techniques for Visible Light Positioning. For this project data is gathered using an experimental setup at Telemic. The setup consists of 4 receivers and 36 LEDs. The LEDs are mounted on the ceiling in a 6x6 grid while the receivers are positioned on the ground. Each receiver can move in a square grid of approximately 1.2m while each receiver is separated over a distance of approximately 1.5m resulting in a total coverage of almost 3m^2. For each position a measurement can be taken giving a 6x6 matrix of the received signal strength of each led. These measurements are then used as input for our machine learning algorithm while the position of the measurement is used as our desired output. The first experiments result in an average accuracy of 1.9cm on the test set.

## Getting Started

The code can be cloned to a local directory by using the standard git command.
```
git clone https://github.com/Maximvda/ML_VLP.git
```
As of now the code cannot be run as the dataset is not yet publicly available. If you still desire to use this code, adaptations will be required to the dataset and preprocessing python files in order to work with a new dataset.  For new datasets it is likely that adaptations will also be required to the network architecture.

### Prerequisites

This code uses the Pytorch framework version 1.3.1 and a couple of python libraries which can all be installed using Conda.

### Usage

The code can be run by running the following command.
```
python main.py
```
Additional arguments can be passed to this command to change some of the behaviour. Descriptions of these extra arguments can be seen by running.
```
python main.py --help
```
Default values of these arguments can be set in [config](https://github.com/Maximvda/ML_VLP/blob/master/utils/config.py).

## Training and running tests

ToDo

## Required features (ToDo)
* Model with only couple of LEDs as input and their ID
* Model trained on only centre LEDs and see how accurate positions are around the centre
* Model with a less dense LED grid

* Simulation to generate training data base on model of the LED

## Authors

* **Maxim Van den Abeele**
* **Jona Beysens** - *Matlab Implementation*
