from dash import dcc, html
import pandas as pd
import pathlib
import dash_bootstrap_components as dbc

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()


def textBox(text, style=None, className=""):
    return html.Div(dcc.Markdown(
        text.replace(
            "  ", ""
        ),
    ),
        className="text-box card-component ",
    )


def text_article(content, style=None):
    list = []
    for i in content:
        if i["type"] == "popup":
            list.append(dcc.Markdown(
                i["text"].replace(
                    "  ", ""
                ),
                id="rid-" + i["text"],
                className="combound popup-trigger"
            ))
            list.append(dbc.Tooltip(
                i["hovtxt"], target="rid-" + i["text"]))
        elif i["type"] == "txt":
            list.append(dcc.Markdown(
                i["text"].replace(
                    "  ", ""
                ),
                className="combound"
            ))
        else:
            list.append(dcc.Markdown(
                i["text"].replace(
                    "  ", ""
                ),
                id="rid-" + i["text"].strip(),
                className="combound refrence",
            )
            )

            list.append(dbc.Tooltip(
                dcc.Markdown(
                    i["hovtxt"].replace(
                        "  ", "",
                    ),
                    link_target="_blank",),
                target="rid-" + i["text"].strip(),
                delay={'show': 0, 'hide': 3000}))
    return html.Div(list,
                    className="text-box card-component",
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
