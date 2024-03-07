from typing import Callable

import pandas as pd
from pages.plots import plot_series, scatter_plot, watter_plot, box_plot, survival_plot, swimmer_plot

from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import output_widget, render_widget  
from shiny.types import ImgData
from htmltools import HTML, div

adlbc = pd.read_csv("raw/adlbc.csv")
adsl = pd.read_csv("raw/adsl.csv")
adae = pd.read_csv("raw/adae.csv")
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
                            width="auto",
                    ),
                    ui.input_select(
                            "param1",
                            "Parameter Category 1:",
                            choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "ALT", width="auto"
                    ),
                    ui.input_select(
                            "param2",
                            "Parameter Category 2:",
                            choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "AST", width="auto"
                    ),
                    ui.div(style = "position: relative;height: 40px;"),
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
    
@module.ui
def scatter_ui():
    return ui.nav_panel(
        "Scatter Plot",
        ui.layout_columns(
            ui.card(
                    ui.input_select(
                            "param1",
                            "Parameter Category 1:",
                            choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "BILI", width="auto"
                    ),
                    ui.input_select(
                            "param2",
                            "Parameter Category 2:",
                            choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "AST", width="auto"
                    ),
                    ui.div(style = "position: relative;height: 40px;"),
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
    
@module.ui
def watter_ui():
    return ui.nav_panel(
        "Watterfall Plot",
        ui.layout_columns(
            ui.card(
                    ui.input_select(
                            "param1",
                            "Parameter Category:",
                            choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "ALT", width="auto"
                    ),
                    ui.div(style = "position: relative;height: 40px;"),
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

@module.ui
def box_ui():
    return ui.nav_panel(
        "Box Plot",
        ui.layout_columns(
            ui.card(
                    ui.input_select(
                            "param1",
                            "Parameter Category:",
                            choices=[value for value in adlbc["paramcd"].unique() if not value.startswith('_')], selected= "SODIUM", width="auto"
                    ),
                    ui.div(style = "position: relative;height: 40px;"),
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
    

@module.ui
def survival_ui():
    return ui.nav_panel(
        "Survival Plot",
            ui.card(
                    output_widget("survival"),
                    style=" height: 900px;  border: none;", class_="survival_card"
        )
)

@module.server
def survival_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    df: Callable[[], pd.DataFrame],
):
    @reactive.Calc()
    def filtered_data() -> pd.DataFrame:
        return df

    @render_widget
    def survival():
        return survival_plot(filtered_data())
    

@module.ui
def swimmer_ui():
    return ui.nav_panel(
        "Swimmer Plot",
        ui.layout_columns(
            ui.card(
                ui.input_select(
                    "decod",
                    "Parameter Category",
                    choices=[str(value) for value in adae["aedecod"].unique() if isinstance(value, str) and not value.startswith('_')],
                    selected="APPLICATION SITE ERYTHEMA",
                    width="auto"
                ),
                ui.div(style="position: relative;height: 40px;"),
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
                output_widget("swimmer"),
                style=" height: 900px;  border: none;"
            ), col_widths=(3, 9)
        )
    )


@module.server
def swimmer_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    df: Callable[[], pd.DataFrame],
):
    @reactive.Calc()
    def filtered_data() -> pd.DataFrame:
        return df

    @render_widget
    def swimmer():
        return swimmer_plot(adae, filtered_data(), input.decod)
    
    
