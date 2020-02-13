# Visible Light Positioning with Machine Learning

The goal of this project is to use machine learning techniques for Visible Light Positioning. For this project data is gathered using an experimental setup at Telemic. The setup consists of 4 receivers and 36 LEDs. The LEDs are mounted on the ceiling in a 6x6 grid while the receivers are positioned on the ground. Each receiver can move in a square grid of approximately 1.2m while each receiver is separated over a distance of approximately 1.5m resulting in a total coverage of almost 9m<sup>2</sup>. For each position a measurement can be taken giving a 6x6 matrix of the received signal strength of each LED. These measurements are then used as input for our machine learning algorithm, while the position of the measurement is used as our desired output. The first experiments result in an average accuracy of 1.9cm on the test set.

## Getting Started

The code can be cloned to a local directory by using the standard git command.
```
git clone https://github.com/Maximvda/ML_VLP.git
```
As of now the code cannot be instantly run as the dataset is not yet publicly available. If you still desire to use this code, adaptations will be required to the dataset and the preprocessing of it. For new datasets it is likely that adaptations will also be required to the network architecture.

### Prerequisites

This code uses the Pytorch framework version 1.3.1 and a couple of python libraries which can all be installed using Conda.

### Usage

A model can be trained, using the default settings by running the following command.
```
python main.py
```
Additional arguments can be passed to this command to change some of the behaviour. Descriptions of these extra arguments can be seen by running.
```
python main.py --help
```
Default values of these arguments can be set in [config](https://github.com/Maximvda/ML_VLP/blob/master/utils/config.py).

### Different TX configurations
One of the arguments allows you to change the TX configuration. This makes it possible to run multiple experiments using the 6x6 LED grid. Choosing a specific configuration means that all the TX which are not selected are set to zero. The different possible configurations that are implemented can be seen in the figure below. If you wish to add a different configuration then that is possible in the [preprocessing](https://github.com/Maximvda/ML_VLP/blob/master/dataset/preprocess.py) file.
<img src="https://github.com/Maximvda/ML_VLP/blob/media/LED_Configuartions.png" width="512">

## Experiments
A couple of predefined experiments can also be run by using the argument --experiment.

### Experiment 1
Experiment one looks at how much performance improvement you get by using more received signals as input for the network. The experiment starts with a model that only uses one signal as input. This signal is the received signal with the highest RSS. More signals are added throughout the experiment each time adding the next best signal, the one with the highest RSS. Of course, when only one signal is used the position estimate will be bad.
The first experiment will start by training a model that only uses the best received signal as input, so only from one TX. A new model is trained after the first one is finished but now using the two best received signals. This is repeated till all the signals are used. This experiment investigates the influence of using increasingly more inputs in the model on the position estimation.

### Experiment 2
The second experiment investigates the performance difference between different TX configurations. A sweep is performed over the 6 different, predefined TX configurations. For each of these a model is trained and the performance of these is evaluated against each other. A plot is made showing the distance on the validation set during the training process. The table (ref) shows the best obtained score on the test set for each model.

## Training and running tests

ToDo

## Required features (ToDo)
- [x] Model with only couple of LEDs as input and their ID
- [x] Model with a less dense LED grid and different configurations
- [x] Create plot of distance for increasingly number of TX being used
- [x] Dynamic model size depending on number of inputs. **Example:** for TX between 26 and 36 -> 6x6 network for TX between 17 and 25 -> 5x5 network
- [ ] Experiment difference between dynamic and fixed model, do it for good TX_config
- [x] Add height to the model
- [x] Check the accuracy in specific area's (square subtracted from square)
- [ ] Experiment training on simulation data and testing on real data
- [ ] Experiment using transfer learning (adding a bit of real data to the simulation data)
- [ ] Experiment investigating the influence of how much data that is used (subsets of the dataset to train)
- [ ] Experiment looking at improvement when a less dense grid is used for the simulation and test. Because now grid is so dense that position is already well known when you know beneath which TX you are.
- [ ] Need to normalise height and make flexible
- [x] Plot heatmaps of error  
- [ ] Train a model on 3 TX and then test this model on again 3 TX in same configuration but different position. If this works then a model can be trained for specific LED configurations and then applied in multiple situations.
- [ ] Make the code flexible to either train to care about the height or not
###
- [x] Simulation to generate training data base on model of the LED
- [ ] Add noise to the simulation
- [ ] Model the PD
- [ ] Kalman filter to estimate position when following a path
### Possible improvements
- [x] Add more layers to network
- [ ] Small network for each output instead of one model for all estimates
- [x] Change loss to BCE(distance,0)
- [ ] Position relative to TX
- [ ] Moving average (Kalman filter or something else) when position is estimated (moving around in a room position will never jump but is continously changing)
- [ ] Only 1 non-linear function tested right now (LeakyReLU)

## Authors

* **Maxim Van den Abeele**
* **Jona Beysens** - *Matlab Implementation*
