from typing import Callable

import pandas as pd
from pages.plots import plot_series, scatter_plot, watter_plot, box_plot

from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import output_widget, render_widget  
from shiny.types import ImgData
from htmltools import HTML, div

adlbc = pd.read_csv("raw/adlbc.csv")
adsl = pd.read_csv("raw/adsl.csv")
adsl.rename(columns={'trt01a': 'trta'}, inplace=True)

@module.ui
def series_ui():
    return ui.nav_panel(
        "Series Plot",
        ui.layout_columns(
            ui.card(
                    ui.input_select(
                            "usubjid",
                            "Subject ID:",
                            choices=[value for value in adlbc["usubjid"].unique() if not value.startswith('_')],
                            width="500px",
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
                ),col_widths=(3,9)
        )
    )

@module.server
def series_server(
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
        return plot_series(filtered_data(), input.usubjid, input.param1, input.param2)
    
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
def scatter_ui():
    return ui.nav_panel(
        "Scatter Plot",
        ui.layout_columns(
            ui.card(
                    ui.input_select(
                            "param1",
                            "Parameter Category 1:",
                            choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "BILI", width="500px"
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
                    output_widget("scatter"),
                    style=" height: 900px;  border: none;"
                ),col_widths=(3,9)
        )
    )

@module.server
def scatter_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    df: Callable[[], pd.DataFrame],
):
    @reactive.Calc()
    def filtered_data() -> pd.DataFrame:
        return df

    @render_widget
    def scatter():
        return scatter_plot(filtered_data(), adsl, input.param1, input.param2)
    
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
def watter_ui():
    return ui.nav_panel(
        "Watterfall Plot",
        ui.layout_columns(
            ui.card(
                    ui.input_select(
                            "param1",
                            "Parameter Category:",
                            choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "ALT", width="500px"
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
                    output_widget("watter"),
                    style=" height: 900px;  border: none;"
                ),col_widths=(3,9)
        )
    )

@module.server
def watter_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    df: Callable[[], pd.DataFrame],
):
    @reactive.Calc()
    def filtered_data() -> pd.DataFrame:
        return df

    @render_widget
    def watter():
        return watter_plot(filtered_data(),input.param1)
    
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
def box_ui():
    return ui.nav_panel(
        "Box Plot",
        ui.layout_columns(
            ui.card(
                    ui.input_select(
                            "param1",
                            "Parameter Category:",
                            choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "SODIUM", width="500px"
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
                    output_widget("box"),
                    style=" height: 900px;  border: none;"
                ),col_widths=(3,9)
        )
    )

@module.server
def box_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    df: Callable[[], pd.DataFrame],
):
    @reactive.Calc()
    def filtered_data() -> pd.DataFrame:
        return df

    @render_widget
    def box():
        return box_plot(filtered_data(),input.param1)
    
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