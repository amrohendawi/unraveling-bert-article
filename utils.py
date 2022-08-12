from dash import dcc, html
import pandas as pd
import pathlib
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()


def add_tooltip(label, button_text, id, href=None):
    return dmc.Tooltip(
        label=label,
        transition="slide-up",
        transitionDuration=300,
        transitionTimingFunction="ease",
        wrapLines=True,
        withArrow=True,
        children=[
            dmc.Button(
                button_text,
                color="gray",
                variant="outline",
                id=id,
            ) if href is None else dmc.Anchor(
                dmc.Button(
                    button_text,
                    color="gray",
                    variant="outline",
                    id=id,
                ),
                href=href,
            )
        ],
    )


def df_to_matrix(df):
    df_matrix = pd.DataFrame(columns=df['sourceTask'].unique())
    for _, row in df.iterrows():
        df_matrix.loc[row['destinationTask'], row['sourceTask']] = row['value']
    return df_matrix


def read_tasks_nested_tables(folder, convert_csvs=lambda x: x):
    new_dict = {}
    for dir in DATA_PATH.joinpath(folder).iterdir():
        # if the directory is a directory, call the function again
        if dir.is_dir():
            new_dict[dir.name] = read_tasks_nested_tables(
                folder + "/" + dir.name, convert_csvs=convert_csvs)
        # if the directory is a csv file, read the file and add the content to the dictionary
        if dir.is_file() and dir.suffix == ".csv":
            # read csv an convert it to json
            df = convert_csvs(pd.read_csv(dir))
            new_dict[dir.name] = df
    return new_dict
