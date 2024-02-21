from typing import Callable

import pandas as pd
from pages.plots import plot_auc_curve, plot_precision_recall_curve, plot_score_distribution

from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import output_widget, render_widget  

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
                    )
            ),
            ui.card(
                    output_widget("series"),
                    style=" height: 900px;  border: none;"
                ),col_widths=(3,9)
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
        return df.loc[df["usubjid"] == input.usubjid()]

    @render_widget
    def series():
        return plot_auc_curve(filtered_data(), input.usubjid, input.param1, input.param2)


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
