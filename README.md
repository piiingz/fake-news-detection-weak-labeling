# Weak Labeling of Fake News Articles with Snorkel and Snuba

## Requirements
Create a virtual environment, then run `pip install -r requirements.txt` to install project dependencies.

### File structure
The following folders need to be created:

data/
1. no1_original 
2. no2_original_split/
    * fake_split 
	* real_split
3. no3_all_features_split/
    * fake_split
	* real_split
4. no4_embeddings_split/
    * fake_split
	* real_split
5. no5_embeddings
6. no6_numerical 
7. testset
8. weak_labeling/
    * analysis
    * confusion_matrix
9. describe
10. sources
11. snuba/
    * goal
    * result


## Folder explanations

1. Full dataset with original dataset - NELA-GT-2019 (csv)
2. Split dataset with original features (csv)
3. Split dataset with all features except word embeddings (pkl)
4. Split dataset with only word embeddings (pkl)
5. Full dataset with only word embeddings (pkl)
6. Full dataset with numerical features and true labels (pkl and csv)
7. Cleaned testset as csv
8. Scores for Snorkel weak labeling systems
9. For dataset description, histograms and boxplots
10. Containing the sources from NELA-GT-2019
11. Scores for Snuba weak labeling system