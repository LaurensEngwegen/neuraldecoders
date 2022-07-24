# Comparison of neural decoding strategies for complex motor tasks using high-density electrocorticography

### Laurens Engwegen

This is the repository for my MSc. Thesis in which different neural decoding strategies were implemented. These strategies were evaluated on two 4-class complex motor tasks performed by different patients. The used data is confidential.

* `preprocessing.py` contains the code for the preprocessing steps and the (possible) feature extraction procedure.
* `trials_creation.py` constructs trials from the preprocessed data.
* In the different `*_Classifier.py` files, the code for training and testing the different neural decoders can be found. The [original implementation of EEGNet](https://github.com/vlawhern/arl-eegmodels) was used.
* `utils.py` and `plots.py` contain utility and plotting functions.
* Performed analyses are defined in `main.py`.
