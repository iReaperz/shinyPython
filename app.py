from pathlib import Path

import pandas as pd
from pages.modules import data_view_server, data_view_ui, training_server, training_ui

from shiny import App, Inputs, Outputs, Session, reactive, ui

adlbc = pd.read_csv("raw/adlbc.csv")
adsl = pd.read_csv("raw/adsl.csv")


app_ui = ui.page_navbar(
    training_ui("tab1"),
    data_view_ui("tab2"),
    header=ui.include_css("assets/styles.css"),
    id="tabs",
    title="Armat Analytics",
)


def server(input: Inputs, output: Outputs, session: Session):
    training_server(id="tab1", df=adlbc)

app = App(app_ui, server)
