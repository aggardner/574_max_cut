Alex Gardner
Sakib Alan
Vincent Gonzales

README_LOCAL

How to run our code:

Prerequisites:
	To run branch_and_cut.py, one needs python and gurobipy library installed.


1) To run this file open up terminal, cd into this current directory and type ‘python branch_and_cuy.py’ in the terminal window. This will start the solver,  loop through all the file instances and identify the max cut. While the objective value and vector will be displayed in the console per file, the objective value and time are also recorded in the output.txt file for verification.

2) Run whichever instances you would like, you can comment or uncomment the graphs you would like to run in a list at the bottom of the python file. It will currently run all of the test instances except for the three largest planar instances. Note that the instances such as d657 and d1291 take the longest, currently at 1.65 and 16.5 hours respectively, you can modify which instances the solver will iterate through in line 515, by modifying the “file_list” to choose which instances to solve. 