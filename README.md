# Visible Light Positioning with Machine Learning
## Modular approach

The goal of this project is to use machine learning techniques for Visible Light Positioning. For this project data is gathered using an experimental setup at Telemic. The setup consists of 4 receivers and 36 LEDs. The LEDs are mounted on the ceiling in a 6x6 grid while the receivers are positioned on the ground. Each receiver can move in a square grid of approximately 1.2x1.2m^<sup>2</sup> while each receiver is separated over a distance of approximately 1.5m resulting in a total coverage of almost 9m<sup>2</sup>. For each position a measurement can be taken giving a 6x6 matrix of the received signal strength of each LED. These measurements are then used as input for our machine learning algorithm, while the position of the measurement is used as our desired output. The experimental setup is graphically shown in the figure below.

<img src="https://github.com/Maximvda/ML_VLP/blob/media/Experimental%20setup.png" width="512">

Questions can be sent to: maximvda123@gmail.com

## Getting Started

The code can be cloned to a local directory by using the standard git command.
```
git clone https://github.com/Maximvda/ML_VLP.git
```
As of now the code cannot be instantly run as the dataset is not yet publicly available. If you still desire to use this code, adaptations will be required to the dataset and the preprocessing of it. For new datasets it is likely that adaptations will also be required to the network architecture.

### Prerequisites

This code uses the Pytorch framework version 1.3.1 and a couple of python libraries which can all be installed using Conda.

### Usage

First of all, you should switch to the modular approach branch to use the unit cell model. This can be done using the following command.
```
git checkout modular_approach
```
A model can then be trained, using the default settings by running the following command.
```
python main.py
```
Additional arguments can be passed to change the behaviour and execution of the code. Descriptions of these extra arguments can be seen by running.
```
python main.py --help
```
Or by looking at the configuration file [config](https://github.com/Maximvda/ML_VLP/blob/modular_approach/utils/config.py) where the default values can be directly changes as well.
Parameters are mainly used to adapt the model architecture and change dataset options.
The model architecture is defined by three parameters: model_type, nf and hidden_layers.
How exactly the model is constructed from these parameters can be seen in the figure below.

<img src="https://github.com/Maximvda/ML_VLP/blob/media/models.png" width="400">

Other parameters mostly optional and used to change behaviour of data processing, training behaviour and system settings.

### Unit cells
Different unit cells can be defined, the code already provides two cells, a 3x3 and a 2x2 unit cell.
It should be easy to add new unit cells by expanding two functions, get_cell_mask and get_cel_center_position.
The cell mask should indicate which TXs need to be used in the unit cell, the mask is then shifted across the entire dataset to find all unit cells.
An optional third function can be expanded, cell_rotation, this is only necessary when the rotational data augmentations are used.
All three functions can be expanded in the [config](https://github.com/Maximvda/ML_VLP/blob/modular_approach/utils/config.py) file.


### Data rotations
The rotations boolean, from the parameters, is used to indicate whether the rotational data augmentations should be used or not.
An example of how the data is rotated for the 3x3 unit cell can be seen in the figure below.
These rotations however do cause difficulties since the used dataset does not use perfect unit cells.
Unit cells in the dataset have slight misalignments causing errors when rotating the data samples and thus increasing the difficulty of the regression task.

<img src="https://github.com/Maximvda/ML_VLP/blob/media/rotation.png" width="256">

## Experiments
One predefined experiment can be run by using the argument --experiment.
For these experiments multiple gpu's and workers can be used to speed up training by training multiple models in parallel.
Using single gpu: --gpu_number=0 for multi gpu: --gpu_number=[0,2,3] which uses gpu 0,2 and 3. Number of workers is set with the --workers argument.

### Experiment 1
Experiment one studies influence of the cell type and data rotations on the amount of blockage.
Two plots are made in this experiment.
The first is shown below and plots the 2D accuracy on the validation set in function of the amount of blockage for both the 3x3 and 2x2 unit cell.
It shows that the 3x3 unit cell almost always outperforms the 2x2 cell.
This is expected as the 3x3 unit cell uses 9 TXs compared to the 4 of the other cell, this redundancy should make it easyer to make accurate predictions.
Interesting enough at high amounts of blockage the 2x2 cell starts outperforming the 3x3 cell.
The reason for this is that the 2x2 cell prediction area is smaller than that of the 3x3 cell, a circle with a radius of 90 cm compared to the 1.25 meter radius.
In the experiment it is assumed that the correct unit cell can be located and therefore the position estimate of the 2x2 cell is already better defined without any intervention of the trained model at all.
The models are not able to reliably predict the position when such high amounts of blockage are introduced and therefore the 2x2 cell starts to outperform the 3x3 cell.
<img src="https://github.com/Maximvda/ML_VLP/blob/media/type_infl_blockage.png" width="512">

The second plot is used to study influence of data rotations on the performance.
Therefore, the 2D accuracy on the validation set is plotted in function of the amount of blockage for both a model that uses rotational data augmentations and one cell that does not.
This plot is shown in the image below, it shows that there is actually a decrease in performance with rotational data augmentations.
As previously mentioned this decrease is likely caused due to the imperfections of the dataset.
<img src="https://github.com/Maximvda/ML_VLP/blob/media/rotation_infl_blockage.png" width="512">

### PBT training
Training a model using the Population Based Training algorithm can be done by setting the parameter --pbt_training to true.
The obtained results however are similar to the obtained results without using PBT.
Below, the three best models and there parameters are listed.

| | Model type | Number of features | Hidden layers | batch size | rotations | 2D / 3D on val |
| --- | --- | --- | --- | --- | --- | --- |
| **Best** | Type 2 | 146 | 6 | 1416 | false | 5.52 / 8.07 cm |
| **Second** | Type 2 | 61 | 6 | 738 | false | 5.56 / 8.06 cm |
| **Third** | Type 2 | 153 | 6 | 581 | false | 5.58 / 8.31 cm |

More interesting though are the smaller models in the population of the PBT algorithm.
These models have very little parameters and still obtain incredible results.
Using these models decreases inference time and thus decreases the computational overhead.

| | Model type | Number of features | Hidden layers | batch size | rotations | 2D / 3D on val |
| --- | --- | --- | --- | --- | --- | --- |
| **Best** | Type 2 | 294 | 2 | 183 | false | 5.70 / 8.22 cm |
| **Second** | Type 1 | 292 | 2 | 494 | false | 5.74 / 7.97 cm |
| **Third** | Type 1 | 292 | 3 | 190 | false | 5.81 / 8.50 cm |

Its also interesting to look at some heatmaps of the positioning accuracy.
In the images below, a heatmap for the entire testbed, a heatmap by a unit cell from the train set and a heatmap by a unit cell from the test set are displayed.
| Entire testbed | Train heatmap | Test heatmap |
| --- | --- | --- |
| <img src="https://github.com/Maximvda/ML_VLP/blob/media/grid.png" width="256"> | <img src="https://github.com/Maximvda/ML_VLP/blob/media/train_map.png" width="256"> | <img  src="https://github.com/Maximvda/ML_VLP/blob/media/test_map.png" width="256"> |


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
* Experiment investigating the influence of how much data is needed for accurate predictions
* Path approximation following a moving device with for example a Kalman filter
* Comparison of model for each position coordinate compared to one model predicting all coordinates
* Combining multiple unit cell predictions

## Authors

* **Maxim Van den Abeele**
* **Jona Beysens** - *Matlab Implementation*
