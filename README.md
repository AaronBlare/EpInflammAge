<div align="center">

# EpImAge

</div>

Repository with source code and Gradio [application]() for EpImAge model.

## Paper
[**EpImAge: a new disease-sensitive clock at the interconnection of immunology and epigenetics**]() by 
[A. Kalyakulina](https://orcid.org/0000-0001-9277-502X),
[I. Yusipov](http://orcid.org/0000-0002-0540-9281),
[C. Franceschi](http://orcid.org/0000-0001-9841-6386),
[A. Moskalev](https://orcid.org/0000-0002-3248-1633),
[M. Ivanchenko](http://orcid.org/0000-0002-1903-7423).


## Abstract
We present a new explainable model of disease-sensitive epigenetic-immunological clock (EpImAge) based on the modern deep neural networks. In the first step, we used our own epigenetic and immunologic data obtained for the same participants to build AI models that can predict levels of 24 cytokines from blood DNA methylation levels. In the second step, we used epigenetic data from open sources (GEO repository) to generate synthetic immunological biomarkers and use them further to build the age estimation model. We collected more than 25 thousand DNAm samples from 72 datasets, for which we obtained cytokines levels using the constructed models. The use of state-of-the-art deep neural networks specialized for tabular data allowed us to achieve good quality metrics for healthy controls: overall MAE of about 7 years and a Pearson correlation of 0.85, as well as sensitivity to diseases from different chapters of ICD-11. Comparison with 33 other epigenetic clock models showed that our model is among the top ones in both criteria - age prediction error and number of detected diseases. The use of explainable AI approaches allows us to investigate the impact of each individual cytokine contributes to the resulting age prediction.

## Project Structure
```
├── app.py                        <- Gradio application hosted on HF
│
├── configs                       <- Pytorch Tabular configs
│   ├── immuno-regression             <- Immunomarkers regression configs
│   └── age-regression                <- Age regression configs
│
├── data                          <- Examples for HF app and data to train models and plot figures
│   ├── examples                      <- Examples for HF app
│   ├── immuno-regression             <- Immunomarkers regression configs
│   └── age-regression                <- Age regression configs
│
├── logs                          <- Directory for training models
│
├── models                        <- Trained models
│   ├── Immunomarkers                 <- Cytokines models
│   │   ├── CXCL9                         <- Model for CXCL9
│   │   ├── CCL11                         <- Model for CCL11
│   │   └── ...                           <- Models for the rest cytokines
│   └── EpImAge                       <- EpImAge model
│
├── notebooks                     <- Jupyter notebooks 
│   ├── 1-immuno-regression           <- Training cytokines models
│   ├── 2-age-regression              <- Training EpImAge
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
├── README.md                     <- This file
│
└── LICENSE                       <- Licence file
```

## Install dependencies

```bash
# clone project
git clone https://github.com/GillianGrayson/EpImAge
cd EpImAge

# [OPTIONAL] create conda environment
conda create -n env_name python=3.11
conda activate env_name

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## License

This repository is licensed under * License.