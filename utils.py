from dash import dcc, html
import pandas as pd
import pathlib
import dash_bootstrap_components as dbc

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()


def textBox(text, style=None, class_name="", text_id="", dangerously_allow_html=False):
    return html.Div(dcc.Markdown(
        text,
        dangerously_allow_html=dangerously_allow_html,
    ),
        style=style,
        className="text-box card-component " + class_name,
        id=text_id,
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
