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

| Model              | Baseline [[log file](https://github.com/yashkant/pythia/blob/add-cot-test/verify_cot/COT/code/log-baseline-run.txt)]  | Author's COT [[log file](https://github.com/yashkant/pythia/blob/add-cot-test/verify_cot/COT/code/log-author-run.txt)] | Pythia's COT [[log file](https://github.com/yashkant/pythia/blob/add-cot-test/verify_cot/COT/code/log-custom-run.txt)]
|:-------------------|:---------------------|:---------------------|:---------------------|
| PreAct ResNet-18                |               5.53%  |              4.92%  |        4.85%

