from pandas import DataFrame
import plotly.express as px
import plotly.graph_objects as go


def plot_score_distribution(df: DataFrame):
    plot = (
        ggplot(df, aes(x="training_score"))
        + geom_density(fill="blue", alpha=0.3)
        + theme_minimal()
        + labs(title="Model scores", x="Score")
    )
    return plot


def plot_auc_curve(df: DataFrame, subjid, first_val, second_val):
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
    print(first_val.get(), second_val.get())
    # Preparing data for further analysis   
    uln_values = filtered_raw.groupby("paramcd").agg(minUALN=("a1hi", "min"), maxUALN=("a1hi", "max")).reset_index()
    alt_min = uln_values.loc[uln_values["paramcd"] == first_val.get(), "minUALN"].values[0]
    ast_min = uln_values.loc[uln_values["paramcd"] == second_val.get(), "minUALN"].values[0]
    print(alt_min,ast_min)

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

    fig.update_layout(height =800, template = "simple_white",
                        legend = dict(orientation = "h", 
                                    title = "", x = 0.5, y = 0.155,
                                    font = dict(size = 12, color = "black"), bordercolor = "black", borderwidth = 1), 
                        title_text=f"<b>{first_val.get()} and {second_val.get()} Results Over Time. (Safety Analysis Set)</b>", title_x=0.535,
                        title_font=dict(size=24, family="Balto"))

    fig.add_trace(go.Table(
        header=dict(values=[], fill=dict(color='rgba(0, 0, 0, 0)')),
        cells=dict(
            values=[table_data[param].round(3) for param in table_data.columns],
            fill=dict(color='rgba(0, 0, 0, 0)')
        ),
        domain=dict(y=[0, 0.1])  # Adjust the y values as needed
    ))

    # Adjust the yaxis domain to leave space for the table
    fig.update_layout(
        yaxis=dict(domain=[0.25, 1])  # Adjust the values as needed
    )

        
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=[str(val) for val in filtered_raw["ady"][filtered_raw["paramcd"] == paramcd].unique()]
        )
    )


    fig.add_annotation(x=1, y=alt_min + 0.21,text=f'{first_val.get()} ULN',showarrow=False,font=dict(size=8),xref="paper")
    fig.add_annotation(x=1,y=ast_min + 0.21, text=f'{second_val.get()} ULN', showarrow=False, font=dict(size=8), xref="paper")
    fig.add_annotation(showarrow=False,xref='paper', x=0, yref='paper',y=0.012,text=f"{second_val.get()}",font=dict(size=12))
    fig.add_annotation(showarrow=False,xref='paper', x=0, yref='paper',y=0.05,text=f"{first_val.get()}",font=dict(size=12))
    fig.add_annotation(showarrow=False,xref='paper', x=0.03, yref='paper',y=0.1,text="Change from Baseline",font=dict(size=14))
    fig.add_annotation(showarrow=False,xref='paper', x=-0.06, yref='paper',y=0.65,textangle=-90,text="Analysis value",font=dict(size=18))
    fig.add_annotation(showarrow=False,xref='paper', x=0.535, yref='paper',y= 0.17,text="Study day relative to treatment start day",font=dict(size=18))
    fig.add_annotation(showarrow=False,xref='paper', x=0.535, yref='paper',y= 1.02,text=f'Usubjid: {subjid.get()}, Treatment: {filtered_raw["trta"].unique()[0]}',font=dict(size=18))

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
