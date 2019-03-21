# Complement Objective Training

This code exists to verify the implementation of COT used in Pythia for multi-label task against the author's implementation.  

Please see the original repository at: https://github.com/henry8527/COT

## Usage
For getting baseline results (without COT)
	
	python main.py --sess Baseline_session
	
For training with Author's Implementation of COT

	python main.py --COT --sess COT_session

For training with Custom (Pythia's) Implementation of COT

	python main.py --COT --sess COT_custom_session --use_custom

## Results

The following table shows the best test errors in a 200-epoch training session using different implementations of COT against the baseline. 

| Model              | Baseline  | Author's COT | Pythia's COT
|:-------------------|:---------------------|:---------------------|:---------------------|
| PreAct ResNet-18                |               5.53%  |              4.92%  |        4.85%
