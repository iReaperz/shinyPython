from pathlib import Path

import pandas as pd
from pages.modules import data_view_server, data_view_ui, training_server, training_ui

from shiny import App, Inputs, Outputs, Session, reactive, ui

df = pd.read_csv(Path(__file__).parent / "raw/scores.csv")
app_ui = ui.page_navbar(
    training_ui("tab1"),
    data_view_ui("tab2"),
    header=ui.include_css(Path(__file__).parent / "assets/styles.css"),
    id="tabs",
    title="Armat Analytics",
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc()
    def filtered_data() -> pd.DataFrame:
        return df.loc[df["account"] == input.account()]

    training_server(id="tab1", df=df)
    data_view_server(id="tab2", df=filtered_data)


app = App(app_ui, server)
