import gradio as gr
from pytorch_tabular import TabularModel
import shap
import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import gradio as gr
from tqdm import tqdm
import plotly.graph_objects as go


dir_root = Path(os.getcwd())

dir_out = f"{dir_root}/out"
if not os.path.exists(dir_out):
   os.makedirs(dir_out)

imms = pd.read_excel(f"{dir_root}/models/Immunomarkers/Immunomarkers.xlsx", index_col='feature').index.values
cpgs = pd.read_excel(f"{dir_root}/models/Immunomarkers/CpGs.xlsx", index_col=0).index.values

models_imms = {}
for imm in (pbar := tqdm(imms)):
    pbar.set_description(f"Loading model for {imm}")
    models_imms[imm] = TabularModel.load_model(f"{dir_root}/models/Immunomarkers/{imm}")

model_age = TabularModel.load_model(f"{dir_root}/models/EpImAge")


with gr.Blocks(theme=gr.themes.Default(), title='EpImAge') as app:
    
    gr.Markdown(
        """
        # EpImAge Calculator
        """
    )
    
    with gr.Row():
        with gr.Column():
            markdown_submit = gr.Markdown(
                """
                ## Submit epigenetics data
                ### Instruction
                - Upload your methylation data for 2228 CpGs from [File](https://github.com/GillianGrayson/EpImAge/tree/main/data/CpGs.xlsx)
                - Your data may also contain `Age` column, in which case the mean absolute error (MAE) and Pearson Correlation Coefficient will be calculated.<br>
                Age acceleration will be calculated as well, scatter will be pltted to compare chronological age and EpImAge.
                - Missing values should be `NA` in the corresponding cells.<br>
                Imputation of missing values will be performed using the KNN method using all methylation data from the [Article]().
                - Data expample for GSE87571: [File](https://github.com/GillianGrayson/EpImAge/tree/main/data/GSE87571.xlsx)
                """,
            )
            input_file = gr.File(label='Methylation Data File', file_count='single', file_types=['.xlsx', 'csv'])
            button_submit = gr.Button("Submit data", variant="primary", interactive=False)
        with gr.Column(min_width=800):
            with gr.Row():
                output_metrics = gr.Text(label='Results', visible=False)
                output_file = gr.File(label='Result File', file_types=['.xlsx'], interactive=False, visible=False)
            with gr.Row():
                plot_results = gr.Plot(label='EpImAge Vs Age', visible=False)
    markdown_explain = gr.Markdown(
        """
        ## Age Acceleration Explanation using XAI
        """,
        visible=False)

    
    def check_size(input):
        curr_file_size = os.path.getsize(input)
        if curr_file_size > 1024 * 1024 * 1024:
            raise gr.Error(f"File exceeds 1 GB limit!")
        else:
            return gr.update(interactive=True)
    
       
    def clear():
        dict_gradio = {
            button_submit: gr.update(interactive=False),
            output_metrics: gr.update(value=None, visible=False),
            output_file: gr.update(value=None, visible=False), 
            plot_results: gr.update(value=None, visible=False)
        }
        return dict_gradio
    
    
    def calc_epimage(input):
        print('Read input data file')
        # Read input data file
        if input.endswith('xlsx'):
            data = pd.read_excel(input, index_col=0)
        elif input.endswith('csv'):
            data = pd.read_csv(input, index_col=0)
        else:
            raise gr.Error(f"Unknown file type!")
        
        print('Check features in input file')
        # Check features in input file
        missed_cpgs = list(set(cpgs) - set(data.columns.values))
        if len(missed_cpgs) > 0:
            raise gr.Error(f"Missed {len(missed_cpgs)} CpGs in the input file!")
        
        print("Models' inference")
        # Models' inference
        for imm in imms:
            data[f"{imm}_log"] = models_imms[imm].predict(data)

        data['EpImAge'] = model_age.predict(data.loc[:, [f"{imm}_log" for imm in imms]])
        data.rename(columns={f"{imm}_log": imm for imm in imms}, inplace=True)
        data['Age Acceleration'] = data['EpImAge'] - data['Age']
        
        data_res = data[['Age', 'EpImAge', 'Age Acceleration'] + list(imms)]
        data_res.to_excel(f'{dir_out}/Result.xlsx', index_label='ID')

        if len(data_res) > 1:
            mae = mean_absolute_error(data['Age'].values, data['EpImAge'].values)
            rho = pearsonr(data['Age'].values, data['EpImAge'].values).statistic
        
        print("Plot scatter")
        # Plot scatter
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.loc[:, 'Age'].values,
                y=data.loc[:, 'EpImAge'].values,
                text=data.index.values,
                hovertext=data.index.values,
                showlegend=False,
                mode='markers',
                marker=dict(
                    size=10,
                    opacity=0.75,
                    line=dict(
                        width=1,
                        color='black'
                    ),
                    color='crimson'
                )
            )
        )
        fig.update_xaxes(
            title_text="Age",
            autorange=True,
            showgrid=False,
            zeroline=False,
            linecolor='black',
            showline=True,
            gridcolor='gainsboro',
            gridwidth=0.05,
            mirror=True,
            ticks='outside',
            titlefont=dict(
                color='black',
                size=12
            ),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(
                color='black',
                size=15
            ),
            exponentformat='e',
            showexponent='all'
        )
        fig.update_yaxes(
            title_text=f"EpImAge",
            scaleanchor="x",
            scaleratio=1,
            autorange=True,
            showgrid=False,
            zeroline=False,
            linecolor='black',
            showline=True,
            gridcolor='gainsboro',
            gridwidth=0.05,
            mirror=True,
            ticks='outside',
            titlefont=dict(
                color='black',
                size=12
            ),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(
                color='black',
                size=15
            ),
            exponentformat='e',
            showexponent='all'
        )
        fig.update_layout(
            template="simple_white",
            # autosize=False,
            width=800,
            height=800,
            # margin=go.layout.Margin(
            #     l=80,
            #     r=20,
            #     b=70,
            #     t=20,
            #     pad=0
            # ),
        )
        
        print('Resulted gradio dict')
        # Resulted gradio dict 
        dict_gradio = {
            output_file: gr.update(value=f'{dir_out}/Result.xlsx', visible=True),
            plot_results: gr.update(value=fig, visible=True),
        }
        if data.shape[0] > 1:
            dict_gradio[output_metrics] = gr.update(value=f"MAE: {mae:.3f}\nPearson Rho: {rho:.3f}", visible=True)
        else:
            dict_gradio[output_metrics] = gr.update(value='Only one sample.\nNo metrics can be calculated.', visible=True)
        
        return dict_gradio
    
    
    button_submit.click(
        fn=calc_epimage,
        inputs=[input_file],
        outputs=[output_file, plot_results, output_metrics]
    )
    input_file.clear(
        fn=clear,
        inputs=[],
        outputs=[button_submit, output_metrics, output_file, plot_results]
    )
    input_file.upload(
        fn=check_size,
        inputs=[input_file],
        outputs=[button_submit]
    )


app.launch()
