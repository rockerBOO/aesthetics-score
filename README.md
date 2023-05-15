# Aesthetic Score

<!--toc:start-->
- [Aesthetic Score](#aesthetic-score)
  - [Install](#install)
  - [Usage](#usage)
    - [File](#file)
    - [Directory](#directory)
    - [CSV](#csv)
  - [Development](#development)
  - [Future](#future)
  - [Contributions](#contributions)
  - [Thanks](#thanks)
<!--toc:end-->

Find the aesthetic score of your images. 

> A linear estimator on top of clip to predict the aesthetic quality of pictures 

## Install

```
git clone aesthetic-score
cd aesthetic-score
```

```bash
python -m venv venv
# In cmd.exe
# venv\Scripts\activate.bat
# In PowerShell
# venv\Scripts\Activate.ps1
# Linux/macOS
# source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### File

Supports a single image or directory of images.

```bash
$ python ae-score.py 42.png
42.png 5.879870414733887
average score: 5.879870414733887
```

### Directory

```bash
$ python ae-score.py outputs/
20230421140324_000010_02_42.png 7.030520439147949
20230421140321_000010_01_42.png 6.584927082061768
20230421140338_000010_07_1268125741.png 6.544317245483398
20230421140335_000010_06_3948340960.png 6.528767108917236
20230421140332_000010_05_1963709028.png 6.520832538604736
20230421140327_000010_03_2661845567.png 6.517449855804443
20230421140329_000010_04_13623292.png 6.31651496887207
20230421140344_000010_09_42.png 6.113379001617432
20230421140319_000010_00_42.png 5.516891002655029
20230421140725_000020_10_42.png 5.190633296966553
20230421140346_000010_10_42.png 5.174572944641113
20230421141439_000040_00_42.png 5.082925319671631
20230421141048_000030_02_42.png 5.081836223602295
20230421141051_000030_03_2661845567.png 5.081359386444092
20230421141109_000030_09_42.png 5.081345558166504
20230421141503_000040_08_42.png 5.081308364868164
20230421141451_000040_04_13623292.png 5.081301212310791
20230421141456_000040_06_3948340960.png 5.081297397613525
20230421141059_000030_06_3948340960.png 5.081272125244141
20230421141509_000040_10_42.png 5.0812506675720215
20230421141448_000040_03_2661845567.png 5.081150054931641
20230421141442_000040_01_42.png 5.081146717071533
20230421141057_000030_05_1963709028.png 5.081067085266113
20230421141453_000040_05_1963709028.png 5.081051349639893
20230421141500_000040_07_1268125741.png 5.080873012542725
20230421141104_000030_07_1268125741.png 5.080867290496826
20230421141046_000030_01_42.png 5.080856800079346
20230421141506_000040_09_42.png 5.08085298538208
20230421141054_000030_04_13623292.png 5.080564022064209
20230421141445_000040_02_42.png 5.080531120300293
20230421141107_000030_08_42.png 5.080453395843506
20230421141043_000030_00_42.png 5.080273628234863
20230421141112_000030_10_42.png 5.080152988433838
20230421140701_000020_01_42.png 5.043121814727783
20230421140341_000010_08_42.png 5.041572093963623
20230421140706_000020_03_2661845567.png 4.73273229598999
20230421140711_000020_05_1963709028.png 4.537281513214111
20230421140723_000020_09_42.png 4.5228352546691895
20230421140658_000020_00_42.png 4.478741645812988
20230421140720_000020_08_42.png 4.4732279777526855
20230421140703_000020_02_42.png 4.472269535064697
20230421140708_000020_04_13623292.png 4.331034183502197
20230421140714_000020_06_3948340960.png 4.285477161407471
20230421140717_000020_07_1268125741.png 4.2842793464660645
average score: 5.227843523025513
```

### CSV 

Save the scores to a CSV file:

```bash
$ python ae-score.py 42.png --save_csv
42.png 5.879870414733887
average score: 5.879870414733887
```

then check for `scores-{timestamp}.csv`

## `ae-filter.py`

Show images that are scored with their score and image to view in the browser. Filter by score range.

![Screenshot 2023-05-12 at 20-30-51 Aesthetics Score Filtering](https://github.com/rockerBOO/aesthetics-score/assets/15027/357bf922-c7f6-418a-9bf4-40ed8cc09a15)

```
python ae-filter.py scores-42.csv --server --port 3456 --images_dir /home/rockerboo/images/
```

Note: ae-filter runs a web server that is not designed for production use. Possibly vulnerable, runs only on localhost and not exposed for security. 

## Help

See the help for all the options.

```
python ae-score.py --help
```

```
python ae-filter.py --help
```

## Changelog

- 2023-05-12 - Add ae-filter.py. Use with `--server` to launch a webserver for your CSV scores file.


## Development


## Future

Working towards building a validation pipeline for fine-tuning training. One component will be checking the aesthetic score.

## Contributions

Open for contributions or integrations into other tooling. Any better predictive models would be appreciative.

## Thanks

- https://github.com/grexzen/SD-Chad
- https://github.com/christophschuhmann/improved-aesthetic-predictor
- https://github.com/LAION-AI/aesthetic-predictor
