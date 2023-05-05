# Aesthetic Score

Find the Aesthetic score of your images. 

> A linear estimator on top of clip to predict the aesthetic quality of pictures 

## Install

```
git clone aesthetic-score
cd aesthetic-score
```

```bash
pip -m venv venv
# In cmd.exe
# venv\Scripts\activate.bat
# In PowerShell
# venv\Scripts\Activate.ps1
# Linux/macOS
# source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Supports a single image or directory of images.

```bash
$ python ae-score.py 42.png
42.png 5.879870414733887
average score: 5.879870414733887
```

Save the scores to a CSV file:

```bash
$ python ae-score.py 42.png --save_csv
42.png 5.879870414733887
average score: 5.879870414733887
```

then check for `scores_{timestamp}.csv`

See the help for all the options.

```
python ae-score.py --help
```

## Development

## Future

Working towards building a validation pipeline for fine-tuning training. One component will be checking the aesthetic score.

## Contributions

Open for contributions or integrations into other tooling. Any better predictive models would be appreciative.

## Thanks

- https://github.com/grexzen/SD-Chad
- https://github.com/christophschuhmann/improved-aesthetic-predictor
- https://github.com/LAION-AI/aesthetic-predictor
