from typing import Callable

import pandas as pd
from pages.plots import plot_auc_curve, plot_precision_recall_curve, plot_score_distribution

from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import output_widget, render_widget  
from shiny.types import ImgData
from htmltools import HTML, div

adlbc = pd.read_csv("raw/adlbc.csv")
adsl = pd.read_csv("raw/adsl.csv")

@module.ui
def training_ui():
    return ui.nav_panel(
        "Series Plot",
        ui.layout_columns(
            ui.card(
                    ui.input_select(
                        "usubjid",
                        "Subject ID",
                        choices=[value for value in adlbc["usubjid"].unique() if not value.startswith('_')],  width="500px"
                    ),
                    ui.input_select(
                        "param1",
                        "Parameter Category 1:",
                        choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "ALT", width="500px"
                    ),
                    ui.input_select(
                        "param2",
                        "Parameter Category 2:",
                        choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "AST", width="500px"
                    ),
                    ui.HTML("<div class ='bottomNav'>       \
                                <p class ='pBottom'>        \
                                    Follow Us:              \
                                    <a href='https://www.google.com' class = 'img'> <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/LinkedIn_icon.svg/108px-LinkedIn_icon.svg.png' width='15' height='15'> </a>\
                                    <a href='https://www.google.com' class = 'img'> <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/900px-Octicons-mark-github.svg.png?20180806170715'  width='15' height='15'> </a>\
                                    <a href='https://www.google.com' class = 'img'> <img src='https://upload.wikimedia.org/wikipedia/commons/e/ec/Medium_logo_Monogram.svg' width='15' height='15'> </a>\
                                </p>                        \
                            </div>")
            ),
            ui.card(
                    output_widget("series"),
                    style=" height: 900px;  border: none;"
                ),col_widths=(2,10)
        )
    )

@module.server
def training_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    df: Callable[[], pd.DataFrame],
):
    @reactive.Calc()
    def filtered_data() -> pd.DataFrame:
        return df

    @render_widget
    def series():
        return plot_auc_curve(filtered_data(), input.usubjid, input.param1, input.param2)
    
    @render.image
    def linked():
        img: ImgData = {"src": str("assets/linked.png"), "width": "20px", "height":"20px"}
        return img
    
    @render.image
    def git():
        img: ImgData = {"src": str("assets/git.png"), "width": "20px", "height":"20px"}
        return img
    
    @render.image
    def med():
        img: ImgData = {"src": str("assets/med.png"), "width": "20px", "height":"20px"}
        return img
    


@module.ui
def data_view_ui():
    return ui.nav_panel(
        "View Data",
        ui.layout_columns(
            ui.value_box(
                title="Row count",
                value=ui.output_text("row_count"),
                theme="primary",
            ),
            ui.value_box(
                title="Mean score",
                value=ui.output_text("mean_score"),
                theme="bg-green",
            ),
            gap="20px",
        ),
        ui.layout_columns(
            ui.card(ui.output_data_frame("data")),
            style="margin-top: 20px;",
        ),
    )


@module.server
def data_view_server(
    input: Inputs, output: Outputs, session: Session, df: Callable[[], pd.DataFrame]
):
    @render.text
    def row_count():
        return df().shape[0]

    @render.text
    def mean_score():
        return round(df()["training_score"].mean(), 2)

    @render.data_frame
    def data():
        return df()
