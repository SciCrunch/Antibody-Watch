# Antibody Specificity Classification

## Requirement
* pytorch >= 0.4.0
* numpy >= 1.13.3
* sklearn
* python 3.6 / 3.7
* pytorch-transformers == 1.2.0

To install requirements, run `pip install -r requirements.txt`.
* For SciBERT models, please visit [allenai/scibert](https://github.com/allenai/scibert) for downloading.
* For ImbalancedDatasetSampler, please visit [ImbalancedDatasetSampler](https://github.com/ufoym/imbalanced-dataset-sampler/tree/4672d71cdd7c7aa9f1924a3d4388a9f464f1e11f). Note that this is their April 20, 2020 version.

## Usage
### Before Training
Please refer to [datasets folder](../datasets/) for more details.

### Training
```sh
python train_k_fold_cross_val.py
```
* All implemented models are listed in [models directory](./models/).
* See [train.py](./train.py) for more training arguments.
* Refer to [train_k_fold_cross_val.py](./train_k_fold_cross_val.py) for k-fold cross validation support.

### Testing
```sh
python infer_example_bert_models.py
```

# References
* Aspect Based Sentiment Analysis, PyTorch Implementations.: https://github.com/songyouwei/ABSA-PyTorch
