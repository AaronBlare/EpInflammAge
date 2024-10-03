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
from sklearn.impute import KNNImputer, SimpleImputer
import shutil


dir_root = Path(os.getcwd())

dir_out = f"{dir_root}/out"
Path(dir_out).mkdir(parents=True, exist_ok=True)

df_imms = pd.read_excel(f"{dir_root}/models/Immunomarkers/Immunomarkers.xlsx", index_col='feature')
imms = df_imms.index.values
imms_log = [f"{f}_log" for f in imms]
cpgs = pd.read_excel(f"{dir_root}/models/Immunomarkers/CpGs.xlsx", index_col=0).index.values

models_imms = {}
for imm in (pbar := tqdm(imms)):
    pbar.set_description(f"Loading model for {imm}")
    models_imms[imm] = TabularModel.load_model(f"{dir_root}/models/Immunomarkers/{imm}")

model_age = TabularModel.load_model(f"{dir_root}/models/EpImAge")

bkgrd_xai = pd.read_pickle(f"{dir_root}/models/background-xai.pkl")
bkgrd_imp = pd.read_pickle(f"{dir_root}/models/background-imputation.pkl")


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

with gr.Blocks(theme=gr.themes.Soft(), title='EpImAge', js=js_func, delete_cache=(3600, 3600)) as app:
    
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
                - Missing values should be `NA` in the corresponding cells.
                - [Imputation](https://scikit-learn.org/stable/modules/impute.html) of missing values can be performed using KNN, Mean, and Median methods with all methylation data from the [Paper]().
                - Data expample for GSE87571: [File](https://github.com/GillianGrayson/EpImAge/tree/main/data/examples/GSE87571.xlsx).
                """,
            )
            input_file = gr.File(label='Methylation Data File', file_count='single', file_types=['.xlsx', 'csv'])
            calc_radio = gr.Radio(choices=["KNN", "Mean", "Median"], value="KNN", label="Imputation method")
            calc_button = gr.Button("Submit data", variant="primary", interactive=False)
        with gr.Column(min_width=800):
            with gr.Row():
                output_file = gr.File(label='Result File', file_types=['.xlsx'], interactive=False, visible=False)
                calc_mae = gr.Text(label='Mean Absolute Error', visible=False)
                calc_rho = gr.Text(label='Pearson Correlation', visible=False)
            with gr.Row():
                calc_plot = gr.Plot(visible=False, show_label=False)
    
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
    
       
    def clear(request: gr.Request):
        dir_to_del = f"{dir_out}/{str(request.session_hash)}"
        if Path(dir_to_del).exists() and Path(dir_to_del).is_dir():
            print(f"Delete cache: {dir_to_del}")
            shutil.rmtree(f"{dir_out}/{str(request.session_hash)}")
        
        dict_gradio = {
            calc_button: gr.update(interactive=False),
            output_file: gr.update(value=None, visible=False), 
            calc_mae: gr.update(value=None, visible=False),
            calc_rho: gr.update(value=None, visible=False),
            calc_plot: gr.update(value=None, visible=False),
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
    
    
    def delete_directory(request: gr.Request):        
        dir_to_del = f"{dir_out}/{str(request.session_hash)}"
        if Path(dir_to_del).exists() and Path(dir_to_del).is_dir():
            print(f"Delete cache: {dir_to_del}")
            shutil.rmtree(f"{dir_out}/{str(request.session_hash)}")
        
    
    def progress_for_calc():
        dict_gradio = {
            output_file: gr.update(value=None, visible=False), 
            calc_mae: gr.update(value=None, visible=False),
            calc_rho: gr.update(value=None, visible=False),
            calc_plot: gr.update(visible=True),
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
    
    
    def progress_for_shap():
        dict_gradio = {
            shap_text_id: gr.update(value=None, visible=False),
            shap_text_age: gr.update(value=None, visible=False),
            shap_text_epimage: gr.update(value=None, visible=False),
            shap_markdown_cytokines: gr.update(value=None, visible=False),
            shap_plot: gr.update(value=None, visible=True),
        }
        return dict_gradio
    
    
    def explain(input, request: gr.Request, progress=gr.Progress()):
        print(f"Read from cache: {dir_out}/{str(request.session_hash)}")
        
        progress(0.0, desc='SHAP values calculation')
        data = pd.read_pickle(f"{dir_out}/{str(request.session_hash)}/data.pkl")
        trgt_id = input
        trgt_age = data.at[trgt_id, 'Age']
        trgt_pred = data.at[trgt_id, 'EpImAge']
        trgt_aa = trgt_pred - trgt_age
        
        n_closest = 200
        data_closest = bkgrd_xai.iloc[(bkgrd_xai['EpImAge'] - trgt_age).abs().argsort()[:n_closest]]
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
    
    
    def calc_epimage(input, request: gr.Request, progress=gr.Progress()):
        print(f"Create cache: {dir_out}/{str(request.session_hash)}")
        Path(f"{dir_out}/{str(request.session_hash)}").mkdir(parents=True, exist_ok=True)
        
        # Read input data file
        progress(0.0, desc='Reading input data file')
        if input[input_file].endswith('xlsx'):
            data = pd.read_excel(input[input_file], index_col=0)
        elif input[input_file].endswith('csv'):
            data = pd.read_csv(input[input_file], index_col=0)
        else:
            raise gr.Error(f"Unknown file type!")
        
        # Check features in input file
        progress(0.2, desc='Checking features in input file')
        missed_cpgs = list(set(cpgs) - set(data.columns.values))
        if len(missed_cpgs) > 0:
            raise gr.Error(f"Missed {len(missed_cpgs)} CpGs in the input file!")
        
        # Imputation of missing values
        imp_method = input[calc_radio]
        data.replace({'NA': np.nan}, inplace=True)
        n_nans = data.isna().sum().sum()
        if n_nans > 0:
            print(f"Imputation of {n_nans} missing values using {imp_method} method")
            progress(0.8, desc=f"Imputation of {n_nans} missing values using {imp_method} method")
            bkgrd_imp.set_index(bkgrd_imp.index.astype(str) + f'_imputation_{imp_method}', inplace=True)
            data_all = pd.concat([data, bkgrd_imp], axis=0, verify_integrity=True)
            if imp_method == "KNN":
                imputer = KNNImputer(n_neighbors=5)
            elif imp_method == 'Mean':
                imputer = SimpleImputer(strategy='mean')
            elif imp_method == 'Median':
                imputer = SimpleImputer(strategy='median')
            data_all.loc[:, cpgs] = imputer.fit_transform(data_all.loc[:, cpgs].values) 
            data.loc[data.index, cpgs] = data_all.loc[data.index, cpgs]
                
        # Models' inference
        progress(0.9, desc="Immunology models' inference")
        for imm in imms:
            data[f"{imm}_log"] = models_imms[imm].predict(data)
        progress(0.95, desc='EpImAge model inference')
        data['EpImAge'] = model_age.predict(data.loc[:, [f"{imm}_log" for imm in imms]])
        data['Age Acceleration'] = data['EpImAge'] - data['Age']
        data.to_pickle(f'{dir_out}/{str(request.session_hash)}/data.pkl')
        
        data_res = data[['Age', 'EpImAge', 'Age Acceleration'] + list(imms_log)]
        data_res.rename(columns={f"{imm}_log": imm for imm in imms}).to_excel(f'{dir_out}/{str(request.session_hash)}/Result.xlsx', index_label='ID')

        if len(data_res) > 1:
            mae = mean_absolute_error(data['Age'].values, data['EpImAge'].values)
            rho = scipy.stats.pearsonr(data['Age'].values, data['EpImAge'].values).statistic
        
        # Plot scatter
        progress(0.98, desc='Plotting scatter')
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
            height=450,
        )
        
        # Resulted gradio dict 
        dict_gradio = {
            output_file: gr.update(value=f'{dir_out}/{str(request.session_hash)}/Result.xlsx', visible=True),
            calc_plot: gr.update(value=fig, visible=True),
            calc_mae: gr.update(value=f"{mae:.3f}", visible=True),
            calc_rho: gr.update(value=f"{rho:.3f}", visible=True) if data.shape[0] > 1 else gr.update(value='Only one sample.\nNo metrics can be calculated.', visible=True),
            shap_markdown_main: gr.update(visible=True),
            shap_dropdown: gr.update(choices=list(data.index.values), value=list(data.index.values)[0], interactive=True, visible=True),
            shap_button: gr.update(visible=True)
        }
        
        return dict_gradio
    
    
    calc_button.click(
        fn=progress_for_calc,
        inputs=[],
        outputs=[output_file, calc_plot, calc_mae, calc_rho, shap_markdown_main, shap_dropdown, shap_button, shap_text_id, shap_text_age, shap_text_epimage, shap_markdown_cytokines, shap_plot]
    )
    calc_button.click(
        fn=calc_epimage,
        inputs={input_file, calc_radio},
        outputs=[output_file, calc_plot, calc_mae, calc_rho, shap_markdown_main, shap_dropdown, shap_button]
    )
    input_file.clear(
        fn=clear,
        inputs=[],
        outputs=[calc_button, calc_mae, calc_rho, output_file, calc_plot, shap_markdown_main, shap_dropdown, shap_button, shap_text_id, shap_text_age, shap_text_epimage, shap_markdown_cytokines, shap_plot]
    )
    input_file.upload(
        fn=check_size,
        inputs=[input_file],
        outputs=[calc_button]
    )
    shap_button.click(
        fn=progress_for_shap,
        inputs=[],
        outputs=[shap_text_id, shap_text_age, shap_text_epimage, shap_markdown_cytokines, shap_plot]
    )
    shap_button.click(
        fn=explain,
        inputs=[shap_dropdown],
        outputs=[shap_text_id, shap_text_age, shap_text_epimage, shap_markdown_cytokines, shap_plot]
    )
    app.unload(delete_directory)


app.launch(show_error=True)
