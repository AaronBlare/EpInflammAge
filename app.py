import gradio as gr
from pytorch_tabular import TabularModel
import shap
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import gradio as gr
from tqdm import tqdm


dir_root = Path(os.getcwd())

imms = pd.read_excel(f"{dir_root}/models/Immunomarkers/Immunomarkers.xlsx", index_col='feature').index.values

models_imms = {}
for imm in (pbar := tqdm(imms)):
    pbar.set_description(f"Loading model for {imm}")
    models_imms[imm] = TabularModel.load_model(f"{dir_root}/models/Immunomarkers/{imm}")
    
model_age = TabularModel.load_model(f"{dir_root}/models/EpImAge")

data = pd.read_excel(f"{dir_root}/data/GSE87571.xlsx")

for imm in (pbar := tqdm(imms)):
    pbar.set_description(f"{imm} inference")
    data[f"{imm}_log"] = models_imms[imm].predict(data)
    
data['EpImAge'] = model_age.predict(data.loc[:, [f"{imm}_log" for imm in imms]])

mae = mean_absolute_error(data['Age'].values, data['EpImAge'].values)
rho = pearsonr(data['Age'].values, data['EpImAge'].values).statistic
print(mae)
print(rho)

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)

demo.launch()