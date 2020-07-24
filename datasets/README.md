# Datasets

The datasets can be downloaded from [zenodo](https://zenodo.org/record/3943128#.Xw00gRPitPY).

## About Datasets
The zip file contains two folders: `RRID` and `antibody_specificity`.
* RRID
    * pair-file.xlsx - This file contains the pairs of snippet from antibody_specificity and snippet from rrid-snippet.xlsx file. 
    * rrid-snippet.xlsx - This file contains snippets of RRID (AB_xxxx).
* antibody_specificity
    * dataset.xlsx - This file contains snippets which were labeled as positive, negative, or neutral.
    * dataset_more_negative.xlsx - This file contains 65 more negative data which were also labeled as positive, negative, or neutral.

## Usage
To use this dataset with this code.
1. Extract zip file to this directory.
2. For ABSA, convert dataset.xlsx and dataset_more_negative.xlsx into csv file with only SNIPPET and label as columns.
3. For RRID Linking, convert pair-file.xlsx into csv file with id, PMID, RRID, Snippet_rdw, SNIPPET_3192antibody, and RRID GT as columns.
4. Before training ABSA models, first cd into utils folder and run
```sh
python convert2seq.py
```