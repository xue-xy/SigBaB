# SigBaB
Branch and Bound for Sigmoid-like Neural Network Verification

SigBaB is a branch and bound verifier for neural networks with sigmoid-like activation functions.

## User Manmul
### Installation
First clone this repository via git as follows:
```bash
git clone https://github.com/xue-xy/SigBaB.git
cd kProp
```
Then install the python dependencies:
```bash
pip install -r requirements.txt
```
### Usage
```bash
python run.py --model <model name> --eps <radius> --bab <bab> --tlimit <time> --batch_size <batch> --device <device>
```
+ `<model>`: the model you want to check.
+ `<eps>`: radius, float between 0 and 1.
+ `<bab>`: whether to use branch and bound, True or False.
+ `<tlimit>`: time limit for each property in seconds.
+ `<batch>`: batch size.
+ `<device>`: device to run the tool, cpu or cuda:0.
