from pathlib import Path

import pandas as pd
from pages.modules import scatter_server, scatter_ui, series_server, series_ui, watter_server, watter_ui

from shiny import App, Inputs, Outputs, Session, reactive, ui

adlbc = pd.read_csv("raw/adlbc.csv")
adsl = pd.read_csv("raw/adsl.csv")
adsl.rename(columns={'trt01a': 'trta'}, inplace=True)


app_ui = ui.page_navbar(
    series_ui("tab1"),
    scatter_ui("tab2"),
    watter_ui("tab3"),
    header=ui.include_css("assets/styles.css"),
    id="tabs",
    title="Armat Analytics",
)


def server(input: Inputs, output: Outputs, session: Session):
    series_server(id="tab1", df=adlbc)
    scatter_server(id="tab2", df= adlbc)
    watter_server(id="tab3", df= adlbc)

app = App(app_ui, server)
