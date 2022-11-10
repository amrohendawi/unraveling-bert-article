# Unraveling BERT's Transferability secrets in a Dash article

In this article, the factors affecting BERT's transferability is through visualizations explained.

This demo of the Dash interactive Python framework is developed by [Plotly](https://plot.ly/).

Dash abstracts away all of the technologies and protocols required to build an interactive web-based application and is a simple and effective way to bind a user interface around your Python code.
To learn more check out this [documentation](https://plot.ly/dash).

## Demo
![demo](screenshots/demo.gif)

## Getting Started

### Running the app locally

First, clone the git repo, then create a virtual environment for installing dependencies.
Feel free to use conda or any other environment manager of your choice.

```
git clone https://github.com/amrohendawi/unraveling-bert-article
cd unraveling-bert-article
python -m venv venv
```

Activate the environment and install the requirements with pip

```
source venv/bin/activate
pip install -r requirements.txt
```

Run the app

```
python app.py
```

### Working on the implementation

Before you push to heroku make sure to update the dependencies in the requirements.txt file in case of new additions.
Heroku will automatically deploy the latest version of the app after every commit to the master branch.

```bash
pip freeze > requirements.txt
```

## Built With

- [Dash](https://dash.plot.ly/) - Main server and interactive components
- [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) - Simplifies the creation of UI components
- [Dash Mantine Components](https://www.dash-mantine-components.com/) - Similar to Bootstrap components but with a different style and more options

## Heroku DevOps

You can push the project to production by simply creating a new app on heroku and connecting it to your github repo within 60 seconds.
