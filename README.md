# HGCNMDA
    This work develops a multi-relational graph convolutional network model, namely HGCNMDA, to perform a MiRNA-Disease Association prediction task. 
    
# Example
    To run HGCNMDA on your data, execute the following command from the project home directory:
	'python main.py'.
	
# Dependencies
    HGCNMDA was implemented with python 3.6.3. To run HGCNMDA, you need these packages:    
    torch (1.5.0)            
    torch-scatter (2.0.5)
    torch-sparse (0.6.7)
    torch-geometric (1.6.1)
    numpy (1.18.5)
    pandas (0.20.3)
    networkx (2.0)
    scikit-learn (0.19.1)

# Input 
    the input files include:
    disease-gene associations, miRNA-gene associations, miRNA-disease associations, disease similarity data, miRNA similarity data, and gene network.

# output
    The AUC of the test data based on HGCNMDA.