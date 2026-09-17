.PHONY: gates train analysis control all

gates:
	python src/gates/step_a_prime_v5_3.py
	python src/gates/step_b_prime_v5_3.py

train:
	python src/training/v5_3_train.py

analysis:
	python src/training/trace_b0.py

control:
	python src/control/g4_permutation.py

all: gates train analysis control
