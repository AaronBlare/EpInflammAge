<div align="center">

# EpInflammAge

</div>

Repository with source code and [Gradio Application](https://huggingface.co/spaces/UNNAILab/EpInflammAge) for EpInflammAge model.

## Paper
**EpInflammAge: Deep Epigenetic-Inflammatory Clock for Disease-Associated Biological Aging** by 
[A. Kalyakulina](https://orcid.org/0000-0001-9277-502X),
[I. Yusipov](http://orcid.org/0000-0002-0540-9281),
[Arseniy Trukhanov](https://orcid.org/0000-0001-6398-6549),
[C. Franceschi](http://orcid.org/0000-0001-9841-6386),
[A. Moskalev](https://orcid.org/0000-0002-3248-1633),
[M. Ivanchenko](http://orcid.org/0000-0002-1903-7423).


## Abstract
We present EpInflammAge, an explainable deep learning tool that integrates epigenetic and inflammatory markers to create a highly accurate, disease-sensitive biological age predictor. This novel approach bridges two key hallmarks of aging - epigenetic alterations and immunosenescence. First, epigenetic and inflammatory data from the same participants was used for AI models predicting levels of 24 cytokines from blood DNA methylation. Second, open-source epigenetic data (25 thousand samples) was used for generating synthetic inflammatory biomarkers and training an age estimation model. Using state-of-the-art deep neural networks optimized for tabular data analysis, EpInflammAge achieves competitive performance metrics against 34 epigenetic clock models, including an overall mean absolute error of 7 years and a Pearson correlation of 0.85 in healthy controls, while demonstrating robust sensitivity across multiple disease categories. Explainable AI revealed the contribution of each feature to the age prediction. The sensitivity to multiple diseases due to combining inflammatory and epigenetic profiles is promising for both research and clinical applications. EpInflammAge is released as an easy-to-use web tool that generates the age estimates and levels of inflammatory parameters for methylation data, with the detailed report on the contribution of input variables to the model output for each sample.


## Project Structure
```
├── app.py                        <- Gradio application hosted on HF
│
├── configs                       <- Pytorch Tabular configs
│   ├── cytokines-regression      <- Inflammatory markers regression configs
│   └── age-regression            <- Age regression configs
│
├── data                          <- Examples for HF app and data to train models and plot figures
│   ├── examples                      <- Examples for HF app
│   ├── cytokines-regression          <- Inflammatory markers regression configs
│   └── age-regression                <- Age regression configs
│
├── logs                          <- Directory for training models
│
├── models                        <- Trained models
│   ├── InflammatoryMarkers           <- Cytokines models
│   │   ├── CXCL9                         <- Model for CXCL9
│   │   ├── CCL11                         <- Model for CCL11
│   │   └── ...                           <- Models for the rest cytokines
│   └── EpInflammAge                  <- EpInflammAge model
│
├── notebooks                     <- Jupyter notebooks 
│   ├── 1-cytokines-regression        <- Training cytokines models
│   ├── 2-age-regression              <- Training EpInflammAge
│   └── 3-plotting                    <- Plotting figures
│
├── plots                         <- Directory for plotted figures
│
├── src                           <- Source code
│   ├── pt                            <- Pytorch Tabular routines
│   └── utils                         <- Utility scripts
│
├── requirements.txt              <- File for installing python dependencies
│
└── README.md                     <- This file
```

## Install dependencies

```bash
# clone project
git clone https://github.com/GillianGrayson/EpInflammAge
cd EpInflammAge

# [OPTIONAL] create conda environment
conda create -n env_name python=3.11
conda activate env_name

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Run Gradio application locally

Gradio Application for EpInflammAge model, which is hosted on [HuggingFace](https://huggingface.co/spaces/UNNAILab/EpInflammAge), can be run locally:

```bash
python app.py
```