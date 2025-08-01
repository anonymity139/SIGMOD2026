Readme (Interactive Regret Query in High Dimension)
=========================
This package contains all source codes for 
a. Algorithm APC 
	1. It is an algorithm designed by us. 
	2. The code is in file highdim.py.
b. Algorithm UH-Random
	1. It is the SOTA algorithm used for comparison. 
	2. The code is in file uh.py.
c. Algorithm UH-Simplex 
	1. It is an existing algorithm used for comparison. 
	2. The code is in folder uh.py.
d. Algorithm SinglePass
	1. It is an existing algorithm used for comparison. 
	2. The code is in folder single_pass.py.

Usage Step
==========
a. Package
	Please install the packages needed by the code, e.g., pytorch, swiglpk, matplotlib, etc. 
	
b. Execution
	The command has several parameters.
	'''
	python main.py (1)Algorithm_name (2)Dataset_name (3)Epsilon (4)Utility_vector_<u[1], u[2], ..., u[d]>
	'''
	E.g., APC 4dSkyline 0.01 0.1 0.1 0.1 0.1

c. Results
	The results will be shown in the console. 

