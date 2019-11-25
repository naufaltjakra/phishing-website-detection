# Project Overview

> This is a Machine Learning Web App Project using
> Random Forest Algorithm to detect Phishing Website.
> The Random Forest Algorithm is build manually meaning
> No sklearn.
> The dataset contains 17 attributes.

## To Run The Program Open Instruction.md

### The latest problem

 - Result function in app.py line 16 still cause error
 - The goal is to display text on result page whether it's Phishing or Not


# Run Instruction (Using Windows 10)
**1. Install All Dependencies**

On console..
```sh
PC/user/kamu> pip install -r requirements.txt
```

**2. Generate new Pickle Model (optional)**

Open random_forest_manual.py file and Uncomment the pickle code on line 296 & 297

Navigate to the folder on console

On console..
```sh
PC/user/kamu> python random_forest_manual.py
```

**3. Setup the Flask Environment**

On CMD..
```sh
PC/user/kamu> set FLASK_APP=app
```

On PowerShell..
```sh
PC/user/kamu> $env:FLASK_APP = "app.py"
PC/user/kamu> $env:FLASK_DEBUG = 1 (This is optional)
```

Or visit env [documentation](https://flask.palletsprojects.com/en/1.0.x/cli/)

**4. Run Flask**

On Console..
```sh
PC/user/loc> flask run
```

**5. Open Browser**

- On url => localhost:5000 (usually)
