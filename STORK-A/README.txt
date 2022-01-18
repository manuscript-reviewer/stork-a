To train, validate, and test STORK-A use the MASTER_STORK-A_CNN.py file. 

Several classifiers can be created depending on the classification (Aneuploid vs Euploid, Complex Aneuploid vs Euploid, Complex Aneuploid vs Single Aneuploid + Euploid) as well as inputs (age, morphological grades, morphokinetics). Based on these, the path in MASTER_STORK-A_CNN.py line 27 should be changed accordingly. 

Each classification and input combination has its own unique input files which can be found in the Data directory. Each classifier has a R script (found in the scripts directory) to create the input files. For ease, the input files have all be created with a sample of 18 data points.

In the scripts directory, the requirements.txt file indicates all required libraries and version to run MASTER_STORK-A_CNN.py. 

The Data directory contains image_IDs.xlsx which include ID's and their associated image file name. The images subdirectory contains images of 18 embryos, captured around 110 hours post-ICSI using time lapse microscopy. PGTA-Data.xlsx contains clinical information for each of the 18 embryos including patient age, morpohkinetic parameters and morphological assessments.

When running  MASTER_STORK-A_CNN.py several outputs will be created, including a saved model and a csv file with results from the test set. 

* If running MASTER_STORK-A_CNN.py on a local CPU, ensure that cuda is turned off.  