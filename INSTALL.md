# Installation and Usage

To run DeepInfer, we highly recommended to use Python 3.7. The current version has been tested on Python 3.7 using a Mac OS (intel). It is recommended to install the same Python virtual environment for the tool. Furthermore, we used bash shell scripts to automate running the codes using different models and datasets. Below are step-by-step instructions to setup environment and run the tool.

### Environment Setup

Follow these steps to create a virtual environment and download the repository from this repository.

1. Download or clone the repository from here and move to the directory using the terminal:

```
cd ReproducibilityPackage/
```

2. Give execution permission using the following command:

```
chmod +x setup.sh
```

3. Run shell script using the following command to create a virtual environment:

```
./setup.sh
```

Please ensure all the following commands (in setup.sh) executed in MacOS (intel) or Linux terminal.
No need to add any new commands in setup.sh:

```
#!/bin/sh

PYTHON_VERSION="3.7"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
source venv/bin/activate
python unseenPredictionDemo.py
```

If required, run the following command to update pip on Python: `python3 -m pip install --upgrade pip`. Alternatively, you can follow the [Python documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to install virtual environment on your machine.

In the setup.sh, ```PYTHON_VERSION="3.7"``` for Python 3.7.x is by default
However, it can be changed for the other version ```PYTHON_VERSION="3.8" ``` for Python 3.8.x


4. To reproduce an example and key results of table 2 and table 3 using our approach, DeepInfer, please follow below instructions.

### Run the DeepInfer tool on an example model

#### Example Buggy DL programs
Navigate to the ExamplePrograms directory `cd ReproducibilityPackage/` with command line. To execute DeepInfer tool using an example model, execute following commands in the terminal

```
source venv/bin/activate
```
**Example with PD1 model:**
```
python unseenPredictionDemo.py
```
The output will show below results which is also displayed in the 2nd row using the PIMA diabetes dataset in Table 2:

```
Total_GroundTruth_Correct: 119
Total_GroundTruth_Incorrect: 34
Total Violation: 192
Total Satisfied: 1032
Total_DeepInfer_Implication_Correct: 108
Total_DeepInfer_Implication_Incorrect: 43
Total Uncertain: 2
```
**Reproduce the key results of table 2 and table 3**

To reproduce the key results of table 2 and table 3 using our approach, DeepInfer, please execute following commands in the terminal which will go the directory Table 2 and Table 3 respectively:

```
./table2.sh
```

```
./table3.sh
```

The output will be stored in the "table2.csv" and "table3.csv" files in the directory "Table 2" and "Table 3" respectively.
