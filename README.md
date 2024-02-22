# ECE/CS 584 (Spring 2024) Verification of Embedded and Cyberphysical systems

## Programming questions in assignment 2

## Pre-requisite
For both the assignments, make sure that you have PyTorch installed. The assignment question were tested on Python 3.9. We recommend that you install PyTorch version >= 2.1.2. For installing PyTorch you can use the following command:

```
pip install torch==2.1.2
```

If you run into any errors, having a look at [their documentation](https://pytorch.org/get-started/locally/).

For technical questions like package installation, you can contact TA Sanil Chawla <schawla7@illinois.edu> for help.

## Problem 3 - MIP/LP formulation

You are required to verify a pre-trained model.  You shall be leveraging the Gurobi Optimizer provided [here](https://pypi.org/project/gurobipy/). This optimizer helps you solve mixed-integer linear and quadratic optimization problems. 

### Prerequisite 
Please install the library for the Gurobi Optimizer from the link provided above or by running - 
```
pip install gurobipy
```
This shall install gurobipy and also grant you a basic license to work on smaller problems like ours.



### Implementation details

1. Starter code has been provided to you, and you need to only make changes in the **mip_assignment.py** file. More specifically you will need to only implement changes in the 5 functions inside the **Verifier** class -
- **def create_inputs(...)**
- **def add_linear_layer(...)**
- **def add_relu_layer(...)**
- **def solve_objectives(...)**
- **def get_verification_objectives(...)**

2. Make sure to fill every TODO section in the python file and do refer the paper above in case of any difficulties.

3. Hints have been provided at every TODO.

### Evaluation
To run the mip_assignment.py file, you need to exeute the command, 

```
python mip_assignment.py data1.pth
```

We have provided a data1.pth that contains training data from the MNIST dataset, and also provided a pre-trained model.pth for the same. We will be evaluating the correctness of your result on a different test_data.pth. The correct results obtained on running the above command are stored in a logs.txt file that can be referred to check the correctness of your implementation.

The logs.txt file contains the logs of the proper evaluation of the problem. You can compare the last few lines of the logs that will mention if success or fail. Note that, we will not evaluate only on the basis of your results completely matching the logs outputs, code will be evaluated for correctness of logic and a decent attempt at implementing the techniques mentioned discussed in the class.

## Problem 4 - Implementing HardTanh for CROWN 

You are required to implement the CROWN algorithm on a new activation function (hardtanh) by applying the techniques we discussed for ReLU during the class.


Starter code has been provided to you, and you only need to make changes in the **hardTanh_question.py**. The code provides a complete implementation of CROWN for ReLU networks. You should read the code first and understand everything before you work on HardTanh.

### Evaluation

By default, the code use ReLU activation function and is fully working:

```
python crown.py data1.pth
```

To test your implementation of the HardTanH, you need to execute the command - 

```
python crown.py -a hardtanh data1.pth
```

Post running the above command, you should be able to see the lower and upper bounds computed for each output of the network.

Note: In both the above commands, you can replace data1.pth by data2.pth to have another sample instance to run your implementations on. We will test your implementation on a unreleased test datapoint.
