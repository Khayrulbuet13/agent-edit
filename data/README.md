# data

The code in this repository is MIT. The files in this folder are derived from the source datasets,
and the terms of each source apply to the files derived from it. No images are shipped.

This folder ships file lists and box annotations. You download the images from their sources
(links below) and put them under a RAW folder. Read the terms of each source before you reuse or
share anything here.

## What is here

- `manifests/<set>.json`: one per set. It holds the class list, the eval list, the train pool and
  the paths of the images and of the annotation file. `images` is relative to RAW. `annotations` is
  relative to this folder.
- `<set>/annotations*.json`: COCO boxes (and masks for `pets-10`) for the images of the set.

Nine sets are loaded by the code. The four `-mix5` sets pair the classes of their base set into
5 invented classes. They use the same images and the same annotation file as the base set.

| set | classes | eval | train pool | annotation file |
| --- | --- | --- | --- | --- |
| `stanford-cars-10` | 10 | 120 | 724 | `stanford-cars-10/annotations_boxmask.json` |
| `cub-10` | 10 | 120 | 469 | `cub-10/annotations_boxmask.json` |
| `pets-10` | 10 | 120 | 1868 | `pets-10/annotations.json` |
| `aircraft-10` | 10 | 120 | 880 | `aircraft-10/annotations_boxmask.json` |
| `stanford-cars-10-mix5` | 5 | 120 | 724 | same as `stanford-cars-10` |
| `cub-10-mix5` | 5 | 120 | 469 | same as `cub-10` |
| `pets-10-mix5` | 5 | 120 | 1868 | same as `pets-10` |
| `aircraft-10-mix5` | 5 | 120 | 880 | same as `aircraft-10` |
| `xview-ships-10` | 10 | 120 | 149 | `xview-ships-10/annotations_boxmask.json` |

Three more manifests are data only: `cdfsod-artaxor`, `cdfsod-dior`, `cdfsod-clipart1k`. They list
the images of three cross-domain sets. The code does not load them, and no annotation files for
them are shipped.

## RAW layout

RAW is a folder you fill. Every `images` path in a manifest is relative to it.

```
RAW/
  cub200/CUB_200_2011/images/              CUB-200-2011, unpacked
  oxford_pets/images/                      Oxford-IIIT Pet, unpacked
  fgvc-aircraft/fgvc-aircraft-2013b/data/images/    FGVC-Aircraft 2013b, unpacked
  stanford-cars-10/images/                 built, see below
  xview-ships-10/images/                   built, see below
```

Tell the code where this folder and RAW are with two environment variables (run this from the
repository root):

```bash
export AGENT_EDIT_DATA=$PWD/data           # manifests/ and the annotation files
export AGENT_EDIT_DATA_RAW=/path/to/RAW    # the folder above
```

The code reads `manifests/<set>.json` from the first folder, and the images from the second.

## Sources

| set | source dataset | download | folder under RAW | derived file here |
| --- | --- | --- | --- | --- |
| `cub-10` | Caltech-UCSD Birds-200-2011 | https://data.caltech.edu/records/20098 (`CUB_200_2011.tgz`) | `cub200/` (the archive holds `CUB_200_2011/`) | `cub-10/annotations_boxmask.json` |
| `pets-10` | Oxford-IIIT Pet | https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz | `oxford_pets/` (the archive holds `images/`) | `pets-10/annotations.json` |
| `aircraft-10` | FGVC-Aircraft 2013b | https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ (`archives/fgvc-aircraft-2013b.tar.gz`) | `fgvc-aircraft/` (the archive holds `fgvc-aircraft-2013b/`) | `aircraft-10/annotations_boxmask.json` |
| `stanford-cars-10` | Stanford Cars, Hugging Face copy `naufalso/stanford_cars` | https://huggingface.co/datasets/naufalso/stanford_cars (4 parquet files in `data/`) | `stanford-cars-10/`, built by `tools/build_stanford_cars10.py` | `stanford-cars-10/annotations_boxmask.json` |
| `xview-ships-10` | xView training set (images and `xView_train.geojson`) | https://challenge.xviewdataset.org/signup (free registration) | `xview-ships-10/`, built by `tools/build_xview_ships10.py` | `xview-ships-10/annotations_boxmask.json` |

`pets-10` uses the official trimaps for its masks, which come from the same site
(`annotations.tar.gz`). The annotation file here already holds the masks, so you only need the images.

### Build the two derived image sets

Run both from the repository root. The scripts are in `tools/`.

Stanford Cars: the 10 classes are cut out of the four parquet files. The script writes the images
and the annotation file. The annotation file it writes is identical to the one in this folder.
It needs `pyarrow`, which is not a dependency of the package, so add it for this one run:

```bash
uv run --no-project --with pyarrow --with pillow --with numpy \
    python tools/build_stanford_cars10.py PARQUET_DIR RAW/stanford-cars-10
```

`PARQUET_DIR` holds the `train-*.parquet` and `test-*.parquet` files from the Hugging Face copy.

xView: the script cuts one JPEG chip per vessel out of the training tiles.

```bash
uv run --no-project --with numpy --with pillow==10.4.0 \
    python tools/build_xview_ships10.py xView_train.geojson TRAIN_TILE_DIR RAW/xview-ships-10
```

`TRAIN_TILE_DIR` is the folder with the training tiles (`*.tif`).

The annotation file it writes is identical to the one in this folder. Run it with Pillow 10.4.0,
as above, to get the exact chips: they are then identical byte for byte. With a newer Pillow
(13.0.0 was tried), 11 of the 269 chips differ in the JPEG pixels by at most 16 levels (out of
255) because the resize filter changed.

## Terms and citations

Quotes are from the pages linked in the last column. Check them again before you reuse the data.

| set | terms as stated by the source | page | cite |
| --- | --- | --- | --- |
| CUB-200-2011 | "We do not own the copyrights to these images. Their use is restricted to non-commercial research and educational purposes." The CaltechDATA record lists `cc-by`. | https://www.vision.caltech.edu/datasets/cub_200_2011/ and https://data.caltech.edu/records/20098 | Wah, Branson, Welinder, Perona, Belongie. The Caltech-UCSD Birds-200-2011 Dataset. Technical report CNS-TR-2011-001, California Institute of Technology, 2011. |
| Oxford-IIIT Pet | Creative Commons Attribution-ShareAlike 4.0 International. "The copyright remains with the original owners of the images." | https://www.robots.ox.ac.uk/~vgg/data/pets/ | Parkhi, Vedaldi, Zisserman, Jawahar. Cats and Dogs. CVPR 2012. |
| FGVC-Aircraft | The images are available "for non-commercial research purposes only". The original authors "retain the copyright of the respective photographs". | https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ | Maji, Kannala, Rahtu, Blaschko, Vedaldi. Fine-Grained Visual Classification of Aircraft. arXiv:1306.5151, 2013. |
| Stanford Cars | The project page did not load when we checked (connection refused), so we found no terms from the authors. The Hugging Face copy has no licence field. Meta-Album lists the "ImageNet License" for the original release, which is for non-commercial research. Treat it as research use only. | https://meta-album.github.io/datasets/CRS.html and https://huggingface.co/datasets/naufalso/stanford_cars | Krause, Stark, Deng, Fei-Fei. 3D Object Representations for Fine-Grained Categorization. 4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), ICCV Workshops, 2013. |
| xView | "under an Attribution-Noncommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license" (terms, section 5.1). Access needs registration. | https://xviewdataset.org/terms.html | Lam, Kuzma, McGee, Dooley, Laielli, Klaric, Bulatov, McCord. xView: Objects in Context in Overhead Imagery. arXiv:1802.07856, 2018. |

Cross-domain sets (manifests only):

| set | terms as stated by the source | page | cite |
| --- | --- | --- | --- |
| ArTaxOr | CC BY-NC-SA 4.0, as listed on the Kaggle page of the dataset | https://www.kaggle.com/datasets/mistag/arthropod-taxonomy-orders-object-detection-dataset | Drange. Arthropod Taxonomy Orders Object Detection Dataset. Kaggle. |
| DIOR | "Both of the two datasets are freely available under the CC BY-NC 4.0 license agreement." | https://gcheng-nwpu.github.io/ | Li, Wan, Cheng, Meng, Han. Object detection in optical remote sensing images: a survey and a new benchmark. ISPRS Journal of Photogrammetry and Remote Sensing 159, 2020. |
| Clipart1k | "Note that these datasets are meant for education and research purposes only." "We do not hold the copyright to the images in the datasets". | https://github.com/naoto0804/cross-domain-detection/blob/master/datasets/README.md | Inoue, Furuta, Yamasaki, Aizawa. Cross-Domain Weakly-Supervised Object Detection through Progressive Domain Adaptation. CVPR 2018. |

The train and eval lists of the cross-domain sets follow the CD-FSOD benchmark where its files
were used (see `notes` in each manifest): Fu et al., Cross-Domain Few-Shot Object Detection via
Enhanced Open-Set Object Detector, arXiv:2402.03094, 2024 (https://github.com/lovelyqian/CDFSOD-benchmark).
