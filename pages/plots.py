from pandas import DataFrame
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import KaplanMeierFitter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def calculate_pchg(group):
    group['pchg'] = 100 * (group['aval'] - group['base']) / group['base']
    group['MaxPchg'] = group['pchg'].max()
    return group

def scatter_plot(df: DataFrame, df2: DataFrame, first_val, second_val):
    if first_val.get() == second_val.get():
        fig = go.Figure()
        # Add a red cross symbol in the middle
        fig.add_annotation(x=2.5, y=2.5, text="\u274C",font=dict(size=100),showarrow=False,)
        fig.add_annotation(x=2.5, y=4, text="Please select two different Parameter Categories ", font=dict(color="#115667", size=24), showarrow=False)
        # Set layout properties
        fig.update_layout(xaxis=dict(visible=False),yaxis=dict(visible=False),plot_bgcolor="white",dragmode=False)
        return fig

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
        fig = go.Figure()
        # Add a red cross symbol in the middle
        fig.add_annotation(x=2.5, y=2.5, text="\u274C",font=dict(size=100),showarrow=False,)
        fig.add_annotation(x=2.5, y=4, text="Please select two different Parameter Categories ", font=dict(color="#115667", size=24), showarrow=False)
        # Set layout properties
        fig.update_layout(xaxis=dict(visible=False),yaxis=dict(visible=False),plot_bgcolor="white",dragmode=False)
        return fig
        

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

def watter_plot(df: DataFrame, trt_selection):
    filtered_data = df[(df['paramcd'] == trt_selection.get()) & (df['saffl'] == "Y") & (df['avisitn'] > 0)]
    grouped_data = filtered_data.groupby(['trta', 'usubjid']).apply(calculate_pchg)

    # Remove the grouping and flatten the multi-level index
    grouped_data = grouped_data.reset_index(drop=True).groupby("usubjid").head(1).dropna(subset=['pchg'])
    grouped_data = grouped_data.reset_index()[["trta", "usubjid", "MaxPchg"]]
    grouped_data['MaxPchg'] = np.where(grouped_data['MaxPchg'] > 100, 100, grouped_data['MaxPchg'])
    grouped_data['xValues'] = grouped_data.groupby('trta').cumcount() + 1
    
    sorted_data = grouped_data.sort_values(by='MaxPchg', ascending=False)
    sorted_data['xValues_sorted'] = range(1, len(sorted_data) + 1)

    master_fig = make_subplots(rows=len(grouped_data["trta"].unique()), cols=1, subplot_titles=grouped_data["trta"].unique(), vertical_spacing=0.05)
    for i, trt in enumerate(grouped_data["trta"].unique()):
        sorted_data = grouped_data[grouped_data["trta"] == trt].sort_values(by='MaxPchg', ascending=False)
        sorted_data['xValues_sorted'] = range(1, len(sorted_data) + 1)

        list_x = list(sorted_data['xValues_sorted'][sorted_data["MaxPchg"] == 100])

        # Add traces
        trace = go.Bar(x=sorted_data['xValues_sorted'], y=sorted_data['MaxPchg'], text=["U" for x in list_x],
                    textposition='outside', insidetextanchor='start',
                    hovertemplate='<b>Subject</b>: %{customdata}<br>' +
                                    '<b>X Value</b>: %{x}<br>' +
                                    '<b>Change</b>: %{y}' +
                                    '<extra></extra>',
                    customdata=sorted_data["usubjid"])

        rect_shape = go.layout.Shape(
            type='rect', xref='x', yref='y',
            x0=0, y0=sorted_data['MaxPchg'].min() - 10, x1=len(sorted_data) + 1, y1=sorted_data['MaxPchg'].max() + 16,
            line={'width': 1, 'color': 'black'}
        )

        master_fig.add_trace(trace, row=i + 1, col=1)
        master_fig.add_shape(rect_shape, row=i + 1, col=1)

        # Update y-axis range
        master_fig.update_yaxes(range=[sorted_data['MaxPchg'].min() - 10, sorted_data['MaxPchg'].max() + 16], row=i + 1, col=1)

    # Update subtitles
    for i, trt in enumerate(grouped_data["trta"].unique()):
        master_fig['layout']['annotations'][i].update(text=f'{trt}')

    # Add text with increased margin
    master_fig.update_layout(height=900, title_text=f"<b>Waterfall Plot of Maximum Post Baseline Percentage Change in {trt_selection.get()}</b>",title_x=0.5,showlegend=False,
                            plot_bgcolor='white', margin=dict(t=70,b=100), title_font=dict(size=20, family="Balto")  # Increase the bottom margin
    )

    master_fig.add_annotation(showarrow=False,xref='paper', x=-0.05, yref='paper',y=0.5,textangle=-90,text="Maximum post baseline percentage change",font=dict(size=14.5))
    master_fig.add_annotation(text="Each bar represents unique subject's maximum percentage change.", x=0, y=-0.05, showarrow=False, xref="paper", yref="paper",font=dict(size=12))
    master_fig.add_annotation(text="If subject's maximum percentage change was greater than 100 percent then the change was displayed as 100 and indicated with the letter U in plot.", 
                            x=0, y=-0.07, showarrow=False, xref="paper", yref="paper",font=dict(size=12))

    return master_fig

def box_plot(df: DataFrame, trt_selection):
    adlbc_filtred = df[(df['paramcd'] == trt_selection.get()) & (df['saffl'] == "Y") & (df['avisitn'] > 0)]        
    adlbc_plot = adlbc_filtred[["aval","avisitn","trta"]].sort_values(by="avisitn").astype({"avisitn":"str"})

    fig = px.box(adlbc_plot, x = "avisitn", y = "aval", color = "trta")

    fig.update_layout(xaxis_title = "Visit", yaxis_title = f"Analysis value: {trt_selection.get()}", template = "simple_white",
                      legend = dict(orientation = "h", 
                                    title = "", x = 0.3, y = -0.1,
                                    font = dict(size = 12, color = "black"), bordercolor = "black", borderwidth = 1), 
                      title_text="<b>Test Results for {} in Each Visit<b>".format(trt_selection.get()), title_x=0.5, title_font=dict(size=20, family="Balto"))
    
    fig.update_traces(hovertemplate= f'<b>{trt_selection.get()}/b>: %{{y}}<br>' +
                                     '<b>Visit Number</b>: %{x}')
    
    return fig

def survival_plot(df:DataFrame):
    data = df[df["saffl"] == "Y"]
    data['cnsr'] = data['cnsr'].replace({0: 1, 1: 0})
    colors = ["darkturquoise", "yellowgreen", "gold"]
    traces = []

    kmf = KaplanMeierFitter()

    fig = go.Figure()
    for i, trt in enumerate(data["trta"].unique()):
        f = data[data["trta"] == trt]
        T = f['aval']
        C = f['cnsr']
        kmf.fit(T, event_observed=C, label=trt)
        traces.append(kmf.survival_function_)

        # Add Kaplan-Meier curve for Placebo group
        fig.add_trace(go.Scatter(x=traces[i].index,
                                y=traces[i][trt],
                                name=trt,
                                line=dict(color=colors[i],shape='hv'),
                                showlegend=True))

        for t, c in zip(T, C):
            if not c:
                fig.add_trace(go.Scatter(x=[t],
                                        y=[traces[i].loc[t, trt]],
                                        mode='markers',
                                        name = trt,
                                        marker=dict(symbol='circle-open', size=8, color=colors[i]),
                                        showlegend=False))

    fig.add_shape(go.layout.Shape(
                type='rect', xref='x', yref='y',
                x0=-10, y0=0, x1=data["aval"].max() + 15, y1=1.1,
                line={'width': 1}
            ))

    fig.update_layout(title='<b>Survival Analysis for Time to First Dermatologic Event</b>',
                    title_x =0.5,
                    title_font=dict(size=20, family="Balto"),
                    margin=dict(t=50,b=100),
                    plot_bgcolor='white',
                    xaxis_title='<b>Time to Demotologic Event or End of Study (days)</b>',
                    yaxis_title='<b>Survival Probability</b>',
                    legend=dict(x=0.997, y = 0.99, xanchor='right', yanchor='top',
                                font = dict(size = 12, color = "black"), bordercolor = "black", borderwidth = 1))

    fig.update_xaxes(showgrid=True,ticks="outside",tickson="boundaries",ticklen=3)
    fig.update_yaxes(showgrid=True,ticks="outside",tickson="boundaries",ticklen=3)

    fig.add_annotation(text="<i><b>Analysis was done using Kaplan-Meier`s estimate.</b></i>", 
                                x=0, y=-0.13, showarrow=False, xref="paper", yref="paper",font=dict(size=11))

    return fig

def swimmer_plot(df: DataFrame, df2: DataFrame, aedecod):
    merged_data = pd.merge(df[(df["aedecod"] == aedecod.get()) & (df["saffl"] == "Y")],
                       df2.loc[:, ["usubjid", "rfpendtc"]],on="usubjid")

    # Удаление дублирующихся строк
    ErthmResp = merged_data.sort_values(by=['usubjid', 'aedecod', 'astdy', 'aendy', 'aesev'], ascending=[True, True, True, False, False]) \
                .drop_duplicates(subset=['usubjid', 'aedecod', 'astdy'], keep='first')

    # Максимальное количество появлений AE по всем субъектам
    NofOccrncs = ErthmResp.sort_values(by='usubjid') \
                    .assign(occrnc=lambda x: x.groupby('usubjid').cumcount() + 1)

    # Для начала инцидента
    ErthmRespStrt = NofOccrncs[['usubjid', 'aedecod', 'rfpendtc', 'trtsdt', 'trta', 'occrnc', 'astdy']] \
                    .pivot_table(index=['usubjid', 'aedecod', 'rfpendtc', 'trtsdt', 'trta'], columns='occrnc', values='astdy', aggfunc='first') \
                    .add_prefix('start_') \
                    .reset_index()  # добавляем reset_index для возврата 'usubjid' в качестве столбца

    # Для конца инцидента
    ErthmRespEnd = NofOccrncs[['usubjid', 'aedecod', 'rfpendtc', 'trtsdt', 'trta', 'occrnc', 'aendy']] \
                    .pivot_table(index=['usubjid', 'aedecod', 'rfpendtc', 'trtsdt', 'trta'], columns='occrnc', values='aendy', aggfunc='first') \
                    .add_prefix('end_') \
                    .reset_index()  # добавляем reset_index для возврата 'usubjid' в качестве столбца

    # Для оценки тяжести инцидента
    ErthmRespSev = NofOccrncs[['usubjid', 'aedecod', 'rfpendtc', 'trtsdt', 'trta', 'occrnc', 'aesev']] \
                    .pivot_table(index=['usubjid', 'aedecod', 'rfpendtc', 'trtsdt', 'trta'], columns='occrnc', values='aesev', aggfunc='first') \
                    .add_prefix('AeSev_') \
                    .reset_index()  # добавляем reset_index для возврата 'usubjid' в качеств

    PrePlot1 = pd.merge(pd.merge(ErthmRespStrt.reset_index(), ErthmRespEnd.reset_index(), on="usubjid", how= "outer"), ErthmRespSev.reset_index(), on="usubjid", how= "outer")
    PrePlot1["end_2"] = np.nan
    cols = ["usubjid", "start", "AEend", "AeSev", "bar_start", "bar_end"]
    PrePlot3 = pd.DataFrame(columns=cols)

    # copying PrePlot1 to keep PrePlot2

    max_occ = NofOccrncs['occrnc'].max()

    for i in range(1, max_occ + 1):
        AEend = f'AEend{i}'
        starts = f'start_{i}'
        ends = f'end_{i}'
        chkEnd = f'chkEnd{i}'
        chk = f'chk{i}'
        endPoint = f'endPoint{i}'

        PrePlot1 = PrePlot1.assign(chkEnd_=(~PrePlot1[ends].isna()).astype(int))
        PrePlot1[AEend] = np.where(~PrePlot1[starts].isna() & PrePlot1[ends].isna(),
                                    (pd.to_datetime(PrePlot1['rfpendtc'].str[:10]) - pd.to_datetime(PrePlot1['trtsdt'])).dt.days + 1,
                                    PrePlot1[ends])
        PrePlot1[endPoint] = np.where(~PrePlot1[starts].isna() & PrePlot1[ends].isna(), 'FILLEDARROW', np.nan)
        PrePlot1[chkEnd] = PrePlot1.groupby('usubjid')['chkEnd_'].cumsum()
        PrePlot1[chk] = np.where(PrePlot1[ends].isna(), 1, np.nan)



    # Extracting columns containing "endPoint" and coalescing them into a single column
    end_point_cols = [col for col in PrePlot1.columns if 'endPoint' in col]
    PrePlot1['barEndPointr'] = PrePlot1[end_point_cols].apply(lambda row: row.dropna().iloc[0], axis=1)

    # Calculating bar_end column
    PrePlot1['bar_start'] = 0
    PrePlot1['bar_end'] = (pd.to_datetime(PrePlot1['rfpendtc'].str[:10]) - pd.to_datetime(PrePlot1['trtsdt'])).dt.days + 1
    PrePlot2 = PrePlot1.copy()
    # Dropping unnecessary columns
    cols_to_drop = [col for col in PrePlot1.columns if 'endPoint' in col or 'chkEnd_' in col]
    PrePlot1 = PrePlot1.drop(columns=cols_to_drop)


    for i in range(1, max_occ + 1):
        AEend = "AEend" + str(i)
        starts = "start_" + str(i)
        AeSev = "AeSev_" + str(i)
        chk = "chk" + str(i)
        
        # Select relevant columns, filter out rows with NaN starts, and rename columns
        temp_df = PrePlot1[['usubjid', starts, AEend, AeSev, chk]].dropna(subset=[starts])
        temp_df = temp_df.rename(columns={starts: 'start', AEend: 'AEend', AeSev: 'AeSev', chk: 'chk'})
        
        # Check if temp_df is not empty before concatenating
        if not temp_df.empty:
            # Append to PrePlot3
            PrePlot3 = pd.concat([PrePlot3, temp_df], ignore_index=True)

    PrePlot3 = pd.merge(PrePlot3[["usubjid", "start", "AEend", "AeSev"]], PrePlot2[["usubjid", "bar_end", "bar_start"]], on="usubjid", how= "left")

    colors = ["#FF7F50", "#998547", "#803009"]
    colorts_triangle = {"MILD" : "yellow", "MODERATE" : "green", "SEVERE": "blue"}

    # Сортируем DataFrame по столбцу 'bar_end' в обратном порядке
    sorted_df = PrePlot2.sort_values(by='bar_end', ascending=False)
    if "endPoint2" in sorted_df.columns:
        sorted_df.loc[(sorted_df["endPoint1"] == "FILLEDARROW") | (sorted_df["endPoint2"] == "FILLEDARROW"), "barEndPointr"] = "FILLEDARROW"
    else:
        sorted_df.loc[(sorted_df["endPoint1"] == "FILLEDARROW"), "barEndPointr"] = "FILLEDARROW"

    # Строим график
    fig = px.bar(sorted_df, x="bar_end", y="usubjid", color="trta", color_discrete_sequence=colors,orientation="h",
        category_orders={"usubjid": sorted_df["usubjid"].unique().tolist()}
    )

    # Добавляем аннотацию
    for i, sev in enumerate(PrePlot3["AeSev"].unique()):
        fig.add_trace(
            go.Scatter( x=PrePlot3["start"][(PrePlot3["AeSev"] == sev)], y=PrePlot3["usubjid"][(PrePlot3["AeSev"] == sev)],
            mode='markers', name=sev, marker=dict(symbol='circle', size=6, color=colorts_triangle[sev]),
            showlegend=True, legendgroup="group",
            legendgrouptitle=dict(text="<b>Severity</b>",font=dict(family="Bold",size = 16)))
    )

    for i, usubjid in enumerate(PrePlot3["usubjid"]):
        ae_end_values = PrePlot3["AEend"][PrePlot3["usubjid"] == usubjid]
        bar_end_values = PrePlot3["bar_end"][PrePlot3["usubjid"] == usubjid]
        if (ae_end_values[i] == bar_end_values[i]):
            size = 0.1
        else:
            size = 6
            
        fig.add_trace(go.Scatter(
            x=ae_end_values, y=[usubjid], mode='markers', name=PrePlot3["AeSev"][PrePlot3["usubjid"] == usubjid].iloc[0],  
            marker=dict(symbol='diamond', size=size, color = colorts_triangle[PrePlot3["AeSev"][PrePlot3["usubjid"] == usubjid].iloc[0]]),
            showlegend=False, legendgroup="group"
        ))
        
    for usubjid in PrePlot3["usubjid"]:
        start_points = PrePlot3.loc[PrePlot3["usubjid"] == usubjid, "start"]
        end_points = PrePlot3.loc[PrePlot3["usubjid"] == usubjid, "AEend"]
        ae_sev = PrePlot3.loc[PrePlot3["usubjid"] == usubjid, "AeSev"].iloc[0]
        for start_point, end_point in zip(start_points, end_points):
            fig.add_trace(go.Scatter(
                x=[start_point, end_point],
                y=[usubjid, usubjid],
                mode='lines',
                name=f'Line for {usubjid}',
                line=dict(color=colorts_triangle[ae_sev], width=1),
                showlegend=False, legendgroup="group"
            ))

    for i, trt in enumerate(sorted_df["trta"].unique()):
        fig.add_trace(go.Scatter(
            x=sorted_df["bar_end"][(sorted_df["trta"] == trt) & (sorted_df["barEndPointr"] == "FILLEDARROW")],
            y=sorted_df["usubjid"][(sorted_df["trta"] == trt) & (sorted_df["barEndPointr"] == "FILLEDARROW")],
            mode='markers', name=trt, marker=dict(symbol='triangle-right', size=22, color=colors[i]),
            showlegend=False
        ))

    fig.update_layout(
        title='<b>Severity and Duration of Application Site Erythema for Each Subject</b>',title_x =0.5,title_font=dict(size=20, family="Balto"),
        xaxis_title="",yaxis=dict(title ="",showticklabels=False, showgrid=True,ticks="outside",tickson="boundaries",ticklen=3),
        xaxis = dict(showgrid=True,ticks="outside",tickson="boundaries",ticklen=3),template = "simple_white",
        legend = dict(title = "<b>Treatment</b>",groupclick="toggleitem",font = dict(size = 12, color = "black",family="Balto"), bordercolor = "black", borderwidth = 1),
        margin=dict(b=100),
    )


    fig.add_annotation(text="<i><b>Each bar represents one Subject in study</b></i>", 
                                    x=0, y=-0.07, showarrow=False, xref="paper", yref="paper",font=dict(size=11))

    return fig

