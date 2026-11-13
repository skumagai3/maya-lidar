# maya-lidar

Semantic segmentation of ancient Maya structures in airborne lidar, in Keras.

The first goal is a baseline on the labeled Chactun dataset (buildings, platforms and
aguadas in the central Yucatan Peninsula), compared against the ECML PKDD 2021
challenge results. The second is applying the trained model to unlabeled NASA G-LiHT
lidar elsewhere on the peninsula.

## Data

The Chactun dataset has 2,094 tiles, each 240 m x 240 m at 0.5 m per pixel. A lidar
tile is 480x480x3 uint8 holding three relief visualizations of the terrain model
(sky-view factor, positive openness and slope). Each of the three classes has its own
binary mask, with 0 marking the feature and 255 the background. Classes overlap, since
buildings often stand on platforms, so the models use one sigmoid output per class.

Tiles 1765-2093 were the challenge test set and are held out here the same way;
train and validation are drawn at random from tiles 0-1764. The authors removed
geolocation and randomized tile numbers to protect the sites from looting, so a
spatial split is not possible. Neighboring tiles are separated by 20 m buffers.

To get the data, either download it with the script, which uses the public Figshare
API:

```bash
python scripts/download_chactun.py --list   # show the files
python scripts/download_chactun.py          # lidar and mask zips, extracted to data/chactun/
```

or download the full bundle (`22202395.zip`) from the Figshare page and extract it:

```bash
python scripts/extract_chactun.py /path/to/22202395.zip
```

On Colab, put `22202395.zip` in `MyDrive/maya-lidar/data/`. The notebooks extract it to
the VM's local disk on first use.

Other lidar sources for the region, and which ones need a login, are listed in
`DATA_SOURCES.md`.

## Running

The notebooks are written for Colab with a GPU. Each one clones and installs this repo,
mounts Google Drive and saves its outputs to `MyDrive/maya-lidar/runs/`. Set
`REPO_URL` in the first cell.

| Notebook | Question |
|---|---|
| `01_explore_chactun` | What do the tiles and labels look like? Checks the mask encoding and class balance per split. |
| `02_unet_scratch` | How well does a plain U-Net trained from scratch do? |
| `03_resnet50_unet` | Does an ImageNet-pretrained ResNet50 encoder help? |
| `04_viewer` | Interactive look at a finished run: sort tiles by error type, toggle layers, move thresholds. No GPU needed. |

Each experiment notebook runs a short smoke test (64 tiles, 2 epochs) before the full
run. Every run writes to `runs/<experiment>/<timestamp>/`:

```
config.yaml          the exact config used
split.json           tile ids in train and validation
normalization.json   per-band mean and std of the training tiles, class counts
log.csv              loss and validation IoU per epoch
best.keras           checkpoint with the lowest validation loss
scores.json          per-class IoU on validation and test, threshold sweep
diagnostics.json     precision-recall curves, object detection by size
val_outputs.npz      validation sigmoid outputs (float16, about 120 MB), read by 04_viewer
figures/             training curves, threshold sweep, IoU bars, PR curves, object sizes,
                     galleries of the worst tiles, detailed tile figures
```

To run locally instead:

```bash
git clone https://github.com/skumagai3/maya-lidar.git && cd maya-lidar
conda env create -f environment.yml
conda activate maya-lidar
pip install -e .
```

## Method

Inputs are standardized per band with the training tiles' mean and standard deviation.
Augmentation is limited to 90-degree rotations and flips. The loss is BCE plus soft
Dice, weighted equally. The reported metric is IoU per class, with true positives,
false positives and false negatives summed over every pixel of the evaluated split
before dividing, as in the challenge. A pixel counts as a class when its sigmoid output
exceeds a per-class threshold chosen on validation; scores at a fixed 0.5 are saved too.

Settings live in `configs/`. Options that are off by default: bottleneck dropout
(`model.dropout`) and frequency-based class weights in the Dice term
(`loss.dice_class_weights: inv_sqrt_frequency`).

For reference, the best challenge entries reached a mean IoU of about 0.83 using lidar
alone.

## Layout

```
configs/              one YAML per experiment
notebooks/            one notebook per experiment
scripts/              download and extraction
src/mayalidar/
  config.py           configs, paths, run folders
  data.py             tile indexing, extraction, official split, tf.data pipeline
  model.py            U-Net, ResNet50 U-Net, loss, IoU metric
  train.py            training from a config
  evaluate.py         IoU, threshold sweep, per-tile scores
  viz.py              figures
  viewer.py           interactive viewer (ipywidgets)
```

## Data credit

The Chactún dataset was produced by the Research Centre of the Slovenian Academy of
Sciences and Arts (ZRC SAZU) and the Jožef Stefan Institute. The airborne laser
scanning was carried out by the National Center for Airborne Laser Mapping (NCALM) in
May 2016 for the Chactún Regional Project, with research authorized by Mexico's
Instituto Nacional de Antropología e Historia (INAH).

Kokalj, Ž., Džeroski, S., Šprajc, I., Štajdohar, J., Draksler, A. and Somrak, M. (2023).
Machine learning-ready remote sensing data for Maya archaeology. Scientific Data 10, 558.
https://doi.org/10.1038/s41597-023-02455-x

Kokalj, Ž., Džeroski, S., Šprajc, I., Štajdohar, J., Draksler, A. and Somrak, M. (2023).
Machine learning-ready remote sensing data for Maya archaeology: masks, ALS data,
Sentinel-1, Sentinel-2. Figshare. https://doi.org/10.6084/m9.figshare.22202395

Kocev, D., Simidjievski, N., Kostovska, A., Dimitrovski, I. and Kokalj, Ž., eds. (2022).
Discover the Mysteries of the Maya: Selected Contributions from the Machine Learning
Challenge and the Discovery Challenge Workshop at ECML PKDD 2021.
https://arxiv.org/abs/2208.03163

## License

The code is released under the MIT License (see `LICENSE`). The Chactún data are not
part of this repository and are covered by the terms on their Figshare record.
