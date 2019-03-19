# Complement Objective Training

This code exists to verify the implementation of COT used in Pythia for multi-label task against the author's implementation.  

Please see the original repository at: https://github.com/henry8527/COT

## Usage
For getting baseline results
	
	python main.py --sess Baseline_session
	
For training via Complement objective  (Author's Implementation)

	python main.py --COT --sess COT_session

For training via Complement objective using Custom Implementation (Pythia's Implementation)

	python main.py --COT --sess COT_session --use_custom