# Visible Light Positioning with Machine Learning

The goal of this project is to use machine learning techniques for Visible Light Positioning. For this project data is gathered using an experimental setup at Telemic. The setup consists of 4 receivers and 36 LEDs. The LEDs are mounted on the ceiling in a 6x6 grid while the receivers are positioned on the ground. Each receiver can move in a square grid of approximately 1.2x1.2m^<sup>2</sup> while each receiver is separated over a distance of approximately 1.5m resulting in a total coverage of almost 9m<sup>2</sup>. For each position a measurement can be taken giving a 6x6 matrix of the received signal strength of each LED. These measurements are then used as input for our machine learning algorithm, while the position of the measurement is used as our desired output. The experimental setup is graphically shown in the figure below.
<img src="https://github.com/Maximvda/ML_VLP/blob/media/Experimental%20setup.png" width="512">

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
Additional arguments can be passed to change the behaviour and execution of the code. Descriptions of these extra arguments can be seen by running.
```
python main.py --help
```
Or by looking at the configuration file [config](https://github.com/Maximvda/ML_VLP/blob/master/utils/config.py) where the default values can be directly changes as well.
The main things that can be changed by the parameters are the model architecture which is defined by three parameters: model_type, nf, hidden_layers. How exactly the model is constructed from these parameters can be seen in the figure below.
<img src="https://github.com/Maximvda/ML_VLP/blob/media/models.png" width="256">

Other parameters are mainly used to change behaviour of data processing, training behaviour and system settings.


### Different TX configurations
One of the arguments allows you to change the TX configuration. This makes it possible to run multiple experiments using the 6x6 LED grid. Choosing a specific configuration means that all the TX which are not selected are set to zero. The different possible configurations that are implemented can be seen in the figure below. If you wish to add a different configuration then that is possible in the [config](https://github.com/Maximvda/ML_VLP/blob/master/dataset/config.py) file by adding it to the get_configuration_dict function.
<img src="https://github.com/Maximvda/ML_VLP/blob/media/LED_Configuartions.png" width="512">

## Experiments
A couple of predefined experiments can also be run by using the argument --experiment.
For these experiments multiple gpu's and workers can be used to speed up training by training multiple models in parallel.
Using single gpu: --gpu_number=0 for multi gpu: --gpu_number=[0,2,3] which uses gpu 0,2 and 3. Number of workers is set with the --workers argument.

### Experiment 1
Experiment one performs a hyperparameter search to find a good model architecture. The best three obtained models and their performance are listed the table below. The default values are also changed in the config file to correspond with the best found model in this experiment.
| | Model type | Number of features | Hidden layers | 2D / 3D on val | 2D / 3D on test |
| --- | --- | --- | --- | --- | --- |
| **Best** |
| **Second** |
| **Third** |

### Experiment 2
The second experiment investigates the performance difference between different TX configurations. A sweep is performed over the 6 different, predefined TX configurations. For each of these a model is trained and the performance of these is evaluated against each other. A plot is made showing the distance on the validation set during the training process. The table below shows the best obtained score on the test set for each model.

| | Config 1| Config 2| Config 3| Config 4| Config 5| Config 6 |
| --- | --- | --- | --- | --- | --- |--- |
| **2D accuracy** | 0.87 cm | 3.4 cm | 41.39 cm | 1.17 cm| 4.37 cm | 11.37 cm |

### Experiment 3
Experiment three looks at how much performance improvement you get by using more received signals as input for the network. The experiment starts with a model that only uses one signal as input. This signal is the received signal with the highest RSS. More signals are added throughout the experiment each time adding the next best signal, the one with the highest RSS. Of course, when only one signal is used the position estimate will be bad. A plot of the obtained accuracies for each of these models is made at the end of the experiment. This result can be seen in the figure below.
<img src="https://github.com/Maximvda/ML_VLP/blob/media/Best_TX_input-1.png" width="256">

### Experiment 4
Experiment four looks at the influence of blockage of the TXs. 10 models are trained in this experiment each time increasing the amount of blockage.
The blockage is introduced by setting the signal to zero of randomly selected TXs. The obtained scores are listed in the table below.

| | Model 1| Model 2| Model 3| Model 4| Model 5|
| --- | --- | --- | --- | --- | --- |--- |
| **# blockage** | 10% | 20% | 30% | 40%| 50% |
| **2D accuracy** | 1.25 cm | 1.52 cm | 1.62 cm | 2.46 cm| 3.46 cm |
| | Model 6| Model 7| Model 8| Model 9| Model 10|
| **# blockage** | 60% | 70% | 80% | 90%| 100% |
| **2D accuracy** | 5.95 cm | 9.48 cm | 17.70 cm | 52.54 cm| 117.22 cm |

### Exeperiment 5
Experiment five performs quick comparisons between a variety of parameters and settings, more information on this experiment in the thesis.

## Training and running tests

Training a model with the default argument values is as simple as running
```
python main.py
```
Evaluting that trained model on the test set is done by setting is_train argument to false example:
```
python main.py --is_train="false"
```
Running an experiment is as simple as passing the experiment number. It might be useful to set number of workers and the gpu_numbers example:
```
python main.py --experiment=2 --gpu_number=[0,1] --workers=4
```

## Interesting additional features (future research)
* Experiment using transfer learning (adding a bit of real data to the simulation data to improve performance)
* Experiment investigating the influence of how much data is needed for accurate predictions
* Refinement of simulation example: add model for the PD, add reflections
* Path approximation following a moving device with for example Kalman filter
* Comparison of model for each position coordinate compared to one model predicting all coordinates
* Combining multiple unit cell predictions

## Authors

* **Maxim Van den Abeele**
* **Jona Beysens** - *Matlab Implementation*
