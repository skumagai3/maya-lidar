# Lidar sources for the Maya lowlands

| Dataset | Labels | Access | Use here |
|---|---|---|---|
| Chactun ML-ready (Kokalj et al. 2023) | buildings, platforms, aguadas | public Figshare API, scripted | training and benchmark |
| G-LiHT, reprocessed by B. Britton (U. Cincinnati) | none | browser download | first unlabeled transects |
| G-LiHT point clouds (NASA LP DAAC) | none | free Earthdata login, then scriptable | more area, own DTM processing |
| Middle Usumacinta, Tabasco (NCALM, OpenTopography) | none | free OpenTopography account | extra Maya-region terrain |
| MREDD+ / CartoData | none | unclear, old portal links may be dead | later |
| NCALM Mayapan and Puuc surveys | none | not public | none |

## Chactun

Figshare record: https://doi.org/10.6084/m9.figshare.22202395. Data descriptor:
https://doi.org/10.1038/s41597-023-02455-x. `scripts/download_chactun.py` downloads the
lidar and mask zips and extracts them; `scripts/extract_chactun.py` handles the full
bundle if it was downloaded by hand. Notebook 01 indexes any `tile_<id>_*.tif` files it
finds under `data/chactun/`.

## G-LiHT, Britton reprocessing

Collection: https://scholar.uc.edu/collections/5q47rq61v. It has five sub-collections:
ground-filtered LAZ files, DEM GeoTIFFs at 33, 50 and 100 cm per pixel, and 33 cm layered
visualizations. Choose a region (South, North2, Centro, Norte, Out, Chiapas, Campeche),
a flight direction (NFI is east-west, GLAS is north-south) and a transect, then use
"Select an Action" and "Download". The files are large.

The 50 cm DEM is the one to start with. It matches Chactun's 0.5 m resolution, so the
path to model inputs is DEM to `rvt` (sky-view factor, positive openness, slope) to
480 px tiles, with no point-cloud processing.

## G-LiHT point clouds

Product page: https://www.earthdata.nasa.gov/data/catalog/lpcloud-gllidarpc-001.
Search: https://search.earthdata.nasa.gov/search?q=C2763264735-LPCLOUD, filtered to
Mexico. G-LiHT's own map: https://glihtdata.gsfc.nasa.gov. A free Earthdata Login
(https://urs.earthdata.nasa.gov) is needed; after that the `earthaccess` package can
search and download. Files are LAS 1.1 with classified ground returns at more than
10 points per square meter. PDAL can grid the ground returns (classification 2) to a
0.5 m DTM.

## Middle Usumacinta

https://opentopography.org/node/3478. About 943 square kilometers of NCALM lidar in
Tabasco, in Maya territory but outside the Yucatan. OpenTopography can return a DTM at a
chosen resolution.

## Matching the Chactun inputs

The model learns from the three visualizations as the Chactun team computed them and
scaled them to 0-255. New inputs should come from the same software (`rvt`, the Relief
Visualization Toolbox, from the same group) with the parameters given in the data
descriptor, and their band histograms should be compared with Chactun's before any
prediction is trusted.
