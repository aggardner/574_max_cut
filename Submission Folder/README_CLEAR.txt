Alex Gardner
Sakib Alan
Vincent Gonzales

README_CLEAR

How to run our code:

Prerequisites:
	To run branch_and_cut.py, one needs python and gurobipy library installed.

1) Upload all the text graph instances and our script branch_and_cut_CLEAR.py to CLEAR. Make sure they are all in the same directory


2) Run the script "gurobi.sh branch_and_cut_CLEAR.py". The script will loop through all the test files, and calculate the optimal tour. Once the tour completes, the optimal solutions will be printed.


3) Run whichever instances you would like, you can comment or uncomment the graphs you would like to run in a list at the bottom of the python file. It will currently run all of the test instances except for the three largest planar instances.