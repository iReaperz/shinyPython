from pandas import DataFrame
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def scatter_plot(df: DataFrame, df2: DataFrame, first_val, second_val):
    if first_val.get() == second_val.get():
        return {
            'data': [],
            'layout': {
                'annotations': [{'text': 'Please choose different values for the two dropdowns.', 'showarrow': False, 'x': 2.5, 'y': 2.5, 'font': {'size': 16,}}],
                'xaxis': {'showline': False,'showgrid': False,'zeroline': False,'showticklabels': False,'title': ''},
                'yaxis': {'showline': False,'showgrid': False,'zeroline': False,'showticklabels': False,'title': ''}
            }
        }

    # Filtering data
    adlbc = df[(df['avisitn'] > 0) & (df['saffl'] == 'Y') & (df['paramcd'].isin([first_val.get(), second_val.get()]))]
    # Calculating number of subjects in each treatment group
    N_Subjs = df2.groupby('trta').size().reset_index(name='Count')

    # Calculating reference lines
    highs_lows = adlbc.sort_values('paramcd').groupby('paramcd').agg({'a1hi': 'min', 'a1lo': 'max'}).reset_index()
    RefLineH1, RefLineH2 = highs_lows['a1hi'].values
    RefLineL1, RefLineL2 = highs_lows['a1lo'].values

    # Calculating maximum results
    MaxRslts1 = adlbc.dropna(subset=['aval']).groupby(['paramcd', 'trta', 'usubjid'],).agg({'aval': 'max'}).reset_index()
    # Merging with the number of subjects
    MaxRslts2 = pd.merge(MaxRslts1, N_Subjs, on = "trta", how='left')
    MaxRslts2['N_trt'] = MaxRslts2['trta'] + "(N=" + MaxRslts2['Count'].astype(str) + ")"
    transposed = MaxRslts2.pivot(index=["usubjid","trta","N_trt"], columns='paramcd', values='aval')
    n_levels = transposed.index.get_level_values('N_trt')

    fig = px.scatter(transposed, x=second_val.get(), y=first_val.get(), facet_col=n_levels, color=n_levels, 
                 facet_col_spacing = 0.001)

    fig.add_hline(y=RefLineH2,line_width=2, line_dash="dash", line_color="gray")
    fig.add_hline(y=RefLineL2,line_width=2, line_dash="dash", line_color="gray")
    fig.add_vline(x=RefLineH1,line_width=2, line_dash="dash", line_color="gray")
    fig.add_vline(x=RefLineL1,line_width=2, line_dash="dash", line_color="gray")

    for i,anno in enumerate(fig['layout']['annotations']):
        anno['text']=[f"{col_name}" for col_name in n_levels.unique()][i]

    fig.update_traces(mode="markers", 
                    hovertemplate= '<b>Subject</b>: %{customdata}<br>' +
                                    f'<b>{second_val.get()}</b>: %{{x}}<br>' +
                                    f'<b>{first_val.get()}</b>: %{{y}}'+
                                    '<extra></extra>',
                    customdata=transposed.index.get_level_values('usubjid').unique())


    fig.update_xaxes(title_text='',ticks='inside',linecolor='black',type="log",mirror=True)
    fig.update_yaxes(title_text='',ticks='inside',linecolor='black',type="log",mirror=True)

    fig.add_annotation(showarrow=False,xref='paper', x=0.5, yref='paper',y=-0.06,text=f'Maximum post baseline {second_val.get()}', font=dict(size=14))
    fig.add_annotation(showarrow=False,xref='paper', x=-0.03, yref='paper',y=0.5,textangle=-90,text=f'Maximum post baseline {first_val.get()}',font=dict(size=14))
    fig.add_annotation(showarrow=False,xref='paper', x=0, yref='paper',y=-0.07,text='Each data point represents a unique subject.')
    fig.add_annotation(showarrow=False,xref='paper', x=0, yref='paper',y=-0.09,text='Logarithmic scaling was used on both X and Y axis.')

    fig.update_layout(plot_bgcolor='white',showlegend=False,
                      title_text=f'<b>Scatter Plot of {first_val.get()} vs {second_val.get()} (Safety Analysis Set)</b>', 
                      title_x=0.5, title_font=dict(size=20, family="Balto"), margin=dict(l=60, t=80))

    return fig


def plot_series(df: DataFrame, subjid, first_val, second_val):
    if first_val.get() == second_val.get():
        return {
            'data': [],
            'layout': {
                'annotations': [{'text': 'Please choose different values for the two dropdowns.', 'showarrow': False, 'x': 2.5, 'y': 2.5, 'font': {'size': 16,}}],
                'xaxis': {'showline': False,'showgrid': False,'zeroline': False,'showticklabels': False,'title': ''},
                'yaxis': {'showline': False,'showgrid': False,'zeroline': False,'showticklabels': False,'title': ''}
            }
        }

    filtered_raw = df[(df["paramcd"].isin([first_val.get(), second_val.get()])) & 
                       (df["avisitn"] >= 0) & (df["saffl"] == "Y")]

    # Preparing data for further analysis   
    uln_values = filtered_raw.groupby("paramcd").agg(minUALN=("a1hi", "min"), maxUALN=("a1hi", "max")).reset_index()
    alt_min = uln_values.loc[uln_values["paramcd"] == first_val.get(), "minUALN"].values[0]
    ast_min = uln_values.loc[uln_values["paramcd"] == second_val.get(), "minUALN"].values[0]

    filtered_raw = filtered_raw[filtered_raw["usubjid"] == subjid.get()]
    table_data = filtered_raw.groupby(["paramcd", "ady"]).sum("chg").reset_index().pivot(index="paramcd", columns="ady", values="chg")
    colors = ["purple", "darkgreen"]
    line_types = ["dash", "longdash"]

    fig = go.Figure()
    for i, paramcd in enumerate(filtered_raw["paramcd"].unique()):
        fig.add_trace(go.Scatter(
            x=filtered_raw["ady"][filtered_raw["paramcd"] == paramcd].astype({"ady":"str"}),
            y=filtered_raw["aval"][filtered_raw["paramcd"] == paramcd],
            marker=dict(color=colors[i]),
            line=dict(dash = line_types[i]),
            name=paramcd
        ))

    fig.add_hline(y=alt_min,line_width=1, line_dash="solid", line_color="gray", opacity=0.7)
    fig.add_hline(y=ast_min,line_width=1, line_dash="solid", line_color="gray", opacity=0.7)
    
    fig.update_layout(  template = "plotly_white",
                        margin=dict(l=60, t=80),
                        legend = dict(orientation = "h", 
                                    title = "", x = 0.5, y = 0.155,
                                    font = dict(size = 12, color = "black"), bordercolor = "black", borderwidth = 1),
                        yaxis=dict(domain=[0.25, 1]),
                        xaxis=dict(tickmode='array', tickvals=[str(val) for val in filtered_raw["ady"][filtered_raw["paramcd"] == paramcd].unique()]),
                        title_text=f"<b>{first_val.get()} and {second_val.get()} Results Over Time. (Safety Analysis Set)</b>", title_x=0.535,
                        title_font=dict(size=20, family="Balto"))

    fig.add_trace(go.Table(
        header=dict(values=[], fill=dict(color='rgba(0, 0, 0, 0)')),
        cells=dict(
            values=[table_data[param].round(3) for param in table_data.columns],
            fill=dict(color='rgba(0, 0, 0, 0)')
        ),
        domain=dict(y=[0, 0.1])  # Adjust the y values as needed
    ))
    
    fig.update_xaxes(showline=True, linewidth=2, linecolor='gray')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='gray')

    fig.add_annotation(x=1, y=alt_min + 0.21,text=f'{first_val.get()} ULN',showarrow=False,font=dict(size=8),xref="paper")
    fig.add_annotation(x=1,y=ast_min + 0.21, text=f'{second_val.get()} ULN', showarrow=False, font=dict(size=8), xref="paper")
    fig.add_annotation(showarrow=False,xref='paper', x=0, yref='paper',y=0.012,text=f"{second_val.get()}",font=dict(size=12))
    fig.add_annotation(showarrow=False,xref='paper', x=0, yref='paper',y=0.05,text=f"{first_val.get()}",font=dict(size=12))
    fig.add_annotation(showarrow=False,xref='paper', x=0.03, yref='paper',y=0.1,text="Change from Baseline",font=dict(size=14))
    fig.add_annotation(showarrow=False,xref='paper', x=-0.05, yref='paper',y=0.65,textangle=-90,text="Analysis value",font=dict(size=18))
    fig.add_annotation(showarrow=False,xref='paper', x=0.55, yref='paper',y= 0.18,text="Study day relative to treatment start day",font=dict(size=18))
    fig.add_annotation(showarrow=False,xref='paper', x=0.535, yref='paper',y= 1.04,text=f'Usubjid: {subjid.get()}, Treatment: {filtered_raw["trta"].unique()[0]}',font=dict(size=16))

    return fig


def plot_precision_recall_curve(df: DataFrame, true_col: str, pred_col: str):
    precision, recall, _ = precision_recall_curve(df[true_col], df[pred_col])

    pr_df = DataFrame({"precision": precision, "recall": recall})

    plot = (
        ggplot(pr_df, aes(x="recall", y="precision"))
        + geom_line(color="darkorange", size=1.5, show_legend=True, linetype="solid")
        + labs(
            title="Precision-Recall Curve",
            x="Recall",
            y="Precision",
        )
        + theme_minimal()
    )

    return plot
