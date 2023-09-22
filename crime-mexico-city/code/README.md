### Contents

- 9 python notebooks
	- DataPreProcessing: runs the data pre-processing (cleaning and merging of data) using the PreProcessing module
	- ShapefilePreProcessing: runs the data pre-processing (cleaning and merging of data) of shapefile data
	- ExploratoryAnalysisCrimeData: explores the crime reports dataset
	- ExploratoryAnalysisDataset: explores the merged dataset with all variables
	- FeatureExtraction: performs feature extraction (lagged variables) for the time series variables
	- ModelTrainingCrimeCounts-np: Trains the models for crime count forecasting and evaluation with raw data
	- ModelTrainingCrimeCounts-pp: Trains the models for crime count forecasting and evaluation with pre-processed data
	- ModelTrainingCrimeHotspots_partition: Trains the models for crime hotspot clustering/partition to obtain the crime classes
	- ModelTrainingCrimeHotspots_classification: Trains the models for crime hotspot prediction and evaluation
	
- 2 python files
	- PreProcessing: Module containing fucntions used in data pre-processing
	- CrimeStandardization: Class that standardizes the crime types in the crime dataset 
	
The notebooks should be run in the order in which they are listed, however, these can be run separately as the
data they use are located in the 'data' folder. The DataPreProcessing and ShapefilePreProcessing notebooks
produce these data files.
