from turtle import width
import gradio as gr
from pytorch_tabular import TabularModel
import shap
import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
import scipy
import scipy.stats
import gradio as gr
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


dir_root = Path(os.getcwd())

dir_out = f"{dir_root}/out"
if not os.path.exists(dir_out):
   os.makedirs(dir_out)

df_imms = pd.read_excel(f"{dir_root}/models/Immunomarkers/Immunomarkers.xlsx", index_col='feature')
imms = df_imms.index.values
imms_log = [f"{f}_log" for f in imms]
cpgs = pd.read_excel(f"{dir_root}/models/Immunomarkers/CpGs.xlsx", index_col=0).index.values

models_imms = {}
for imm in (pbar := tqdm(imms)):
    pbar.set_description(f"Loading model for {imm}")
    models_imms[imm] = TabularModel.load_model(f"{dir_root}/models/Immunomarkers/{imm}")

model_age = TabularModel.load_model(f"{dir_root}/models/EpImAge")

bkgrd = pd.read_pickle(f"{dir_root}/models/Background.pkl")


def predict_func(X):
    X_df = pd.DataFrame(data=X, columns=imms_log)
    y = model_age.predict(X_df)['Age_prediction'].values
    return y


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(theme=gr.themes.Soft(), title='EpImAge', js=js_func) as app:
    
    gr.Markdown(
        """
        # EpImAge Calculator
        ## Submit epigenetics data
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                ### Instruction
                - Upload your methylation data for 2228 CpGs from [File](https://github.com/GillianGrayson/EpImAge/tree/main/data/CpGs.xlsx).
                - The first column must be a sample ID.
                - Your data must contain `Age` column for metrics (MAE and Pearson Rho) and Age Acceleration calculation.
                - Missing values should be `NA` in the corresponding cells.<br>
                Imputation of missing values will be performed using the KNN method with all methylation data from the [Paper]().
                - Data expample for GSE87571: [File](https://github.com/GillianGrayson/EpImAge/tree/main/data/GSE87571.xlsx).
                """,
            )
            input_file = gr.File(label='Methylation Data File', file_count='single', file_types=['.xlsx', 'csv'])
            button_submit = gr.Button("Submit data", variant="primary", interactive=False)
        with gr.Column(min_width=800):
            with gr.Row():
                output_file = gr.File(label='Result File', file_types=['.xlsx'], interactive=False, visible=False)
                output_mae = gr.Text(label='Mean Absolute Error', visible=False)
                output_rho = gr.Text(label='Pearson Correlation', visible=False)
            with gr.Row():
                plot_results = gr.Plot(label='EpImAge Vs Age', visible=False, show_label=False)
    
    shap_markdown_main = gr.Markdown(
        """
        ## Age Acceleration Explanation using XAI
        """,
        visible=False)
    
    with gr.Row():
        with gr.Column():
            shap_dropdown = gr.Dropdown(label='Choose a sample to get an explanation of the EpImAge prediction', filterable=True, visible=False)
            shap_button = gr.Button("Get explanation", variant="primary", visible=False)
            with gr.Row():
                shap_text_id = gr.Text(label='Sample', visible=False)
                shap_text_age = gr.Text(label='Age', visible=False)
                shap_text_epimage = gr.Text(label='EpImAge', visible=False)
            shap_markdown_cytokines = gr.Markdown(
                """
                ### Most important cytokines:
                """,
                visible=False
            )
        with gr.Column(min_width=800):
            shap_plot = gr.Plot(label='Explanation', visible=False, show_label=False)

    
    def check_size(input):
        curr_file_size = os.path.getsize(input)
        if curr_file_size > 1024 * 1024 * 1024:
            raise gr.Error(f"File exceeds 1 GB limit!")
        else:
            return gr.update(interactive=True)
    
       
    def clear():
        data = pd.DataFrame()
        dict_gradio = {
            button_submit: gr.update(interactive=False),
            output_file: gr.update(value=None, visible=False), 
            output_mae: gr.update(value=None, visible=False),
            output_rho: gr.update(value=None, visible=False),
            plot_results: gr.update(value=None, visible=False),
            shap_markdown_main: gr.update(visible=False),
            shap_dropdown: gr.update(value=None, visible=False),
            shap_button: gr.update(visible=False),
            shap_text_id: gr.update(value=None, visible=False),
            shap_text_age: gr.update(value=None, visible=False),
            shap_text_epimage: gr.update(value=None, visible=False),
            shap_markdown_cytokines: gr.update(visible=False),
            shap_plot: gr.update(value=None, visible=False),
        }
        return dict_gradio
    
    
    def make_visible_results():
        dict_gradio = {
            output_file: gr.update(value=None, visible=False), 
            output_mae: gr.update(value=None, visible=False),
            output_rho: gr.update(value=None, visible=False),
            plot_results: gr.update(visible=True),
            shap_markdown_main: gr.update(visible=False),
            shap_dropdown: gr.update(value=None, visible=False),
            shap_button: gr.update(visible=False),
            shap_text_id: gr.update(value=None, visible=False),
            shap_text_age: gr.update(value=None, visible=False),
            shap_text_epimage: gr.update(value=None, visible=False),
            shap_markdown_cytokines: gr.update(visible=False),
            shap_plot: gr.update(value=None, visible=False),
        }
        return dict_gradio
    
    
    def make_visible_shap():
        dict_gradio = {
            shap_text_id: gr.update(value=None, visible=False),
            shap_text_age: gr.update(value=None, visible=False),
            shap_text_epimage: gr.update(value=None, visible=False),
            shap_markdown_cytokines: gr.update(value=None, visible=False),
            shap_plot: gr.update(value=None, visible=True),
        }
        return dict_gradio
    
    
    def explain(input, progress=gr.Progress()):
        progress(0.0, desc='SHAP values calculation')
        data = pd.read_pickle(f'{dir_out}/data.pkl')
        trgt_id = input
        trgt_age = data.at[trgt_id, 'Age']
        trgt_pred = data.at[trgt_id, 'EpImAge']
        trgt_aa = trgt_pred - trgt_age
        
        n_closest = 200
        data_closest = bkgrd.iloc[(bkgrd['EpImAge'] - trgt_age).abs().argsort()[:n_closest]]
        explainer = shap.SamplingExplainer(predict_func, data_closest.loc[:, imms_log])
        shap_values = explainer.shap_values(data.loc[[trgt_id], imms_log].values)[0]
        shap_values = shap_values * (trgt_pred - trgt_age) / (trgt_pred - explainer.expected_value)
        
        df_less_more = pd.DataFrame(index=imms, columns=['Less', 'More'])
        for f in df_less_more.index:
            df_less_more.at[f, 'Less'] = round(scipy.stats.percentileofscore(data_closest.loc[:, f"{f}_log"].values, data.at[trgt_id, f"{f}_log"]))
            df_less_more.at[f, 'More'] = 100.0 - df_less_more.at[f, 'Less']

        df_shap = pd.DataFrame(index=imms, data=shap_values, columns=[trgt_id])
        df_shap.sort_values(by=trgt_id, key=abs, inplace=True)
        df_shap['cumsum'] = df_shap[trgt_id].cumsum()
        
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, shared_xaxes=False, column_widths=[2.5, 1], horizontal_spacing=0.05, row_titles=[''])
        fig.add_trace(
            go.Waterfall(
                hovertext=["Chrono Age", "EpImAge"],
                orientation="h",
                measure=['absolute', 'relative'],
                y=[-1.5, df_shap.shape[0] + 0.5],
                x=[trgt_age, trgt_aa],
                base=0,
                text=[f"{trgt_age:0.2f}", f"+{trgt_aa:0.2f}" if trgt_aa > 0 else f"{trgt_aa:0.2f}"],
                textposition = "auto",
                decreasing = {"marker":{"color": "deepskyblue", "line": {"color": "black", "width": 1}}},
                increasing = {"marker":{"color": "crimson", "line": {"color": "black", "width": 1}}},
                totals= {"marker":{"color": "dimgray", "line": {"color": "black", "width": 1}}},
                connector={
                    "mode": "between",
                    "line": {"width": 1, "color": "black", "dash": "dot"},
                },
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Waterfall(
                hovertext=df_shap.index.values,
                orientation="h",
                measure=["relative"] * len(imms),
                y=list(range(df_shap.shape[0])),
                x=df_shap[trgt_id].values,
                base=trgt_age,
                text=[f"+{x:0.2f}" if x > 0 else f"{x:0.2f}" for x in df_shap[trgt_id].values],
                textposition = "auto",
                decreasing = {"marker":{"color": "lightblue", "line": {"color": "black", "width": 1}}},
                increasing = {"marker":{"color": "lightcoral", "line": {"color": "black", "width": 1}}},
                connector={
                    "mode": "between",
                    "line": {"width": 1, "color": "black", "dash": "solid"},
                },
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(
            row=1,
            col=1,
            automargin=True,
            tickmode="array",
            tickvals=[-1.5] + list(range(df_shap.shape[0])) + [df_shap.shape[0] + 0.5],
            ticktext=["Chrono Age"] + df_shap.index.to_list() + ["EpImAge"],
            tickfont=dict(size=16),
        )
        fig.update_xaxes(
            row=1,
            col=1,
            automargin=True,
            title='Age',
            titlefont=dict(size=20),
            range=[
                trgt_age - df_shap['cumsum'].abs().max() * 1.25,
                trgt_age + df_shap['cumsum'].abs().max() * 1.25
            ],
        )
        fig.update_traces(row=1, col=1, showlegend=False)

        fig.add_trace(
            go.Bar(
                hovertext=df_shap.index.values,
                orientation="h",
                name='Less',
                x=df_less_more.loc[df_shap.index.values, 'Less'],
                y=list(range(df_shap.shape[0])),
                marker=dict(color='steelblue', line=dict(color="black", width=1)),
                text=df_less_more.loc[df_shap.index.values, 'Less'],
                textposition='auto'
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                hovertext=df_shap.index.values,
                orientation="h",
                name='More',
                x=df_less_more.loc[df_shap.index.values, 'More'],
                y=list(range(df_shap.shape[0])),
                marker=dict(color='violet', line=dict(color="black", width=1)),
                text=df_less_more.loc[df_shap.index.values, 'More'],
                textposition='auto',
            ),
            row=1,
            col=2
        )
        fig.update_xaxes(
            row=1,
            col=2,
            automargin=True,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
        )
        fig.update_yaxes(
            row=1,
            col=2,
            automargin=True,
            showgrid=False,
            showline=False,
            zeroline=False,
            showticklabels=False,
        )
        
        fig.update_layout(barmode="relative")
        fig.update_layout(
            legend=dict(
                title=dict(text="Immunomarkers disribution<br>in samples with same age", side="top"),
                orientation="h",
                yanchor="bottom",
                y=0.95,
                xanchor="center",
                x=0.84
            ),
        )

        fig.update_layout(
            template="none",
            width=800,
            height=800,
        )
        
        # Resulted gradio dict 
        dict_gradio = {
            shap_text_id: gr.update(value=input, visible=True),
            shap_text_age: gr.update(value=f"{trgt_age:.3f}", visible=True),
            shap_text_epimage: gr.update(value=f"{trgt_pred:.3f}", visible=True),
            shap_markdown_cytokines: gr.update(
                value="### Most important cytokines:\n" + '\n'.join(df_imms.loc[df_shap.index.values[:-4:-1], 'Text'].to_list()),
                visible=True
            ),
            shap_plot: gr.update(value=fig, visible=True),
        }
        return dict_gradio
    
    
    def calc_epimage(input, progress=gr.Progress()):
        
        progress(0.0, desc='Reading input data file')
        # Read input data file
        if input.endswith('xlsx'):
            data = pd.read_excel(input, index_col=0)
        elif input.endswith('csv'):
            data = pd.read_csv(input, index_col=0)
        else:
            raise gr.Error(f"Unknown file type!")
        
        progress(0.6, desc='Checking features in input file')
        # Check features in input file
        missed_cpgs = list(set(cpgs) - set(data.columns.values))
        if len(missed_cpgs) > 0:
            raise gr.Error(f"Missed {len(missed_cpgs)} CpGs in the input file!")
        
        # Models' inference
        progress(0.7, desc="Immunology models' inference")
        for imm in imms:
            data[f"{imm}_log"] = models_imms[imm].predict(data)
        progress(0.8, desc='EpImAge model inference')
        data['EpImAge'] = model_age.predict(data.loc[:, [f"{imm}_log" for imm in imms]])
        data['Age Acceleration'] = data['EpImAge'] - data['Age']
        data.to_pickle(f'{dir_out}/data.pkl')
        
        data_res = data[['Age', 'EpImAge', 'Age Acceleration'] + list(imms_log)]
        data_res.rename(columns={f"{imm}_log": imm for imm in imms}).to_excel(f'{dir_out}/Result.xlsx', index_label='ID')

        if len(data_res) > 1:
            mae = mean_absolute_error(data['Age'].values, data['EpImAge'].values)
            rho = scipy.stats.pearsonr(data['Age'].values, data['EpImAge'].values).statistic
        
        progress(0.9, desc='Plotting scatter')
        # Plot scatter
        fig = make_subplots(rows=1, cols=2, shared_yaxes=False, shared_xaxes=False, column_widths=[5, 3], horizontal_spacing=0.15)
        min_plot_age = data[["Age", "EpImAge"]].min().min()
        max_plot_age = data[["Age", "EpImAge"]].max().max()
        shift_plot_age = max_plot_age - min_plot_age
        min_plot_age -= 0.1 * shift_plot_age
        max_plot_age += 0.1 * shift_plot_age
        fig.add_trace(
            go.Scatter(
                x=[min_plot_age, max_plot_age],
                y=[min_plot_age, max_plot_age],
                showlegend=False,
                mode='lines',
                line = dict(color='black', width=2, dash='dot')
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                name='Scatter',
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
            ),
            row=1,
            col=1
        )
        fig.update_xaxes(
            row=1,
            col=1,
            automargin=True,
            title_text="Age",
            autorange=False,
            range=[min_plot_age, max_plot_age],
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
                size=20
            ),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(
                color='black',
                size=16
            ),
            exponentformat='e',
            showexponent='all'
        )
        fig.update_yaxes(
            row=1,
            col=1,
            automargin=True,
            title_text=f"EpImAge",
            # scaleanchor="x",
            # scaleratio=1,
            autorange=False,
            range=[min_plot_age, max_plot_age],
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
                size=20
            ),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(
                color='black',
                size=16
            ),
            exponentformat='e',
            showexponent='all'
        )
        fig.add_trace(
            go.Violin(
                y=data.loc[:, 'Age Acceleration'].values,
                hovertext=data.index.values,
                name="Violin",
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
                line_color='black',
                fillcolor='crimson',
                marker=dict(color='crimson', line=dict(color='black', width=0.5), opacity=0.75),
                points='all',
                bandwidth=np.ptp(data.loc[:, 'Age Acceleration'].values) / 32,
                opacity=0.75
            ),
            row=1,
            col=2
        )
        fig.update_yaxes(
            row=1,
            col=2,
            automargin=True,
            title_text="Age Acceleraton",
            autorange=True,
            showgrid=False,
            zeroline=True,
            linecolor='black',
            showline=True,
            gridcolor='gainsboro',
            gridwidth=0.05,
            mirror=True,
            ticks='outside',
            titlefont=dict(
                color='black',
                size=20
            ),
            showticklabels=True,
            tickangle=0,
            tickfont=dict(
                color='black',
                size=16
            ),
            exponentformat='e',
            showexponent='all'
        )
        fig.update_xaxes(
            row=1,
            col=2,
            automargin=True,
            autorange=False,
            range=[-0.5, 0.3],
            showgrid=False,
            showline=True,
            zeroline=False,
            showticklabels=False,
            mirror=True,
            ticks='outside',
            tickvals=[],

        )
        fig.update_layout(
            template="simple_white",
            width=800,
            height=500,
        )
        
        # Resulted gradio dict 
        dict_gradio = {
            output_file: gr.update(value=f'{dir_out}/Result.xlsx', visible=True),
            plot_results: gr.update(value=fig, visible=True),
            output_mae: gr.update(value=f"{mae:.3f}", visible=True),
            output_rho: gr.update(value=f"{rho:.3f}", visible=True) if data.shape[0] > 1 else gr.update(value='Only one sample.\nNo metrics can be calculated.', visible=True),
            shap_markdown_main: gr.update(visible=True),
            shap_dropdown: gr.update(choices=list(data.index.values), value=list(data.index.values)[0], interactive=True, visible=True),
            shap_button: gr.update(visible=True)
        }
        
        return dict_gradio
    
    
    button_submit.click(
        fn=make_visible_results,
        inputs=[],
        outputs=[output_file, plot_results, output_mae, output_rho, shap_markdown_main, shap_dropdown, shap_button, shap_text_id, shap_text_age, shap_text_epimage, shap_markdown_cytokines, shap_plot]
    )
    button_submit.click(
        fn=calc_epimage,
        inputs=[input_file],
        outputs=[output_file, plot_results, output_mae, output_rho, shap_markdown_main, shap_dropdown, shap_button]
    )
    input_file.clear(
        fn=clear,
        inputs=[],
        outputs=[button_submit, output_mae, output_rho, output_file, plot_results, shap_markdown_main, shap_dropdown, shap_button, shap_text_id, shap_text_age, shap_text_epimage, shap_markdown_cytokines, shap_plot]
    )
    input_file.upload(
        fn=check_size,
        inputs=[input_file],
        outputs=[button_submit]
    )
    shap_button.click(
        fn=make_visible_shap,
        inputs=[],
        outputs=[shap_text_id, shap_text_age, shap_text_epimage, shap_markdown_cytokines, shap_plot]
    )
    shap_button.click(
        fn=explain,
        inputs=[shap_dropdown],
        outputs=[shap_text_id, shap_text_age, shap_text_epimage, shap_markdown_cytokines, shap_plot]
    )


app.launch(show_error=True)
