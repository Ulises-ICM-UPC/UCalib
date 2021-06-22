
# UCalib

`UCalib` is an open source software written in Python for automatic image calibration of camera orientation from a set of manually calibrated images.

### Description
The calibration algorithm assumes that the camera position and the intrinsic parameters of the camera remains unchanged. The result of the process is a common position and intrinsic camera parameters for all images, and the orientation of the cameras for each of the images. The development of this software is suitable for Argus-type video monitoring stations. Details about the algorithm and methodology are described in
> *Simarro, G.; Calvete, D.; Soutoa, P. UCalib: Cameras autocalibration on coastal videomonitoring systems. Submitted to Remote Sens. 2021*

The automatic calibration process consists of two steps:
 1. [Manual calibration of the basis](#basis-calibration)
 2. [Automatic image calibration](#automatic-image-calibration)

### Requirements and project structure
To run the software it is necessary to have Python (3.8) and install the following dependencies:
- cv2 (4.2.0)
- numpy (1.19.5)
- scipy (1.6.0)

In parenthesis we indicate the version with which the software has been tested. It is possible that it works with older versions. 

The structure of the project is the following:
* **`basis`**
  * `basisImage01cal.txt`
  * `basisImage01cdg.txt`
  * `basisImage01cdh.txt`
  * `basisImage01.png`
  * . . .
* **`imagesToAutoCalibrate`**
  * `image000001cal.txt`
  * `image000001.png`
  * . . .
* **`ucalib`**
  * `ucalib.py`

The local modules of `UCalib` are located in the **`ucalib`** folder.

To run a demo with the images in folder **`basis`** and **`imagesToAutoCalibrate`** using a Jupyter Notebook we provide the file `example_notebook.ipynb`. For experienced users, the `example.py` file can be run in a terminal. 

## Basis calibration
To manually calibrate the images selected for the basis, placed in the folder **`basis`**,  it is necessary that each image `<basisImage>.png` is supplied with a file containing the Ground Control Points (GCP) and, optionally, the Horizon Points (HP) together with the water surface level. The structure of each of these files is the following:
* `<basisImage>cdg.txt`: For each GCP one line with 
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`
* `<basisImage>cdh.txt`: For each HP one line with
>`pixel-column`, `pixel-row`
* `<basisImage>zms.txt`: If HP are provided, at the video station, a line with
> `mean-sea-level`

Quantities must be separated by at least one blank space between them and the last record should not be continued with a newline (return).

### Run basis calibration
Import `ucalib` module:


```python
from ucalib import ucalib
```

Set the folder path where files for calibrating the base are located:


```python
pathFolderBasis = 'basis'
```

Set the value of the basis calibration parameter:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Critical reprojection pixel error | `eCrit` | _5._ | _pixel_ |


Run the calibration algorithm of the basis:


```python
eCrit=5.
```

Run the calibration algorithm of the basis:


```python
ucalib.CalibrationOfBasis(pathFolderBasis,eCrit)
```

As a result of the calibration, the calibration file `<basisImage>cal.txt` is generated in the **`basis`** directory for each of the images. This file contains the following parameters:

| Magnitudes | Variables | Units |
|:--|:--:|:--:|
| Camera position coordinates | `xc`, `yc`, `zc` | _m_ |
| Camera orientation angles | `ph`, `sg`, `ta` | _rad_ |
| Lens radial distortion (parabolic, quartic) | `k1a`, `k2a` | _-_ |
| Lens tangential distortion (parabolic, quartic) | `p1a`, `p2a` | _-_ |
| Pixel size | `sc`, `sr` | _-_ |
| Decentering | `oc`, `rr` | _pixel_ |
| Image size | `nc`, `nr` | _pixel_ |
| Calibration error | `errorT`| _pixel_ |

The different calibration files `*cal.txt` differ only in the angles of the camera orientation  (`ph`, `sg`, `ta`) and the calibration error (`errorT`).

In case that for a certain image `<basisImage>.txt` the reprojection error of a GCP is higher than the error E, a message will appear suggesting to modify the values or to delete the point in the file `<basisImage>cdg.txt`. 

## Automatic image calibration

In this second step, each of the images in the folder **`imagesToAutoCalibrate`** will be automatically calibrated. Set the folder path where images to calibrate automatically are stored:


```python
pathFolderImagesToAutoCalibrate = 'imagesToAutoCalibrate'
```

Set the values of the automatic image calibration parameters:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Number of features to identify with ORB | `nORB` | _10000_ | _-_ |
| Critical homography error | `fC` | _5._ | _pixel_ |
| Critical number of pairs | `KC` | _4_ | _-_ |



```python
nORB = 10000
fC, KC = 5., 4
```

Run the algorithm to calibrate images automatically:


```python
ucalib.AutoCalibrationOfImages(pathFolderBasis,pathFolderImagesToAutoCalibrate,nORB,fC,KC)
```

For each of the images `<image>.png` in directory **`imagesToAutoCalibrate`**, a calibration file `<image>cal.txt` with the same characteristics as the one described above will be obtained.
The self-calibration process may fail because the homography error is higher than the one set by the parameter fC, the number of pairs is lower than the critical value KC or ORF cannot identify pairs in the image. In any of these cases it is reported that the image  `<image>.png` is _not automatically calibratable_.

## Contact us

Are you experiencing problems? Do you want to give us a comment? Do you need to get in touch with us? Please contact us!

To do so, we ask you to use the [Issues section](https://github.com/Ulises-ICM-UPC/UCalib/issues) instead of emailing us.

## Contributions

Contributions to this project are welcome. To do a clean pull request, please follow these [guidelines](https://github.com/MarcDiethelm/contributing/blob/master/README.md)

## License

UCalib is released under a [GPLv3 license](https://github.com/Ulises-ICM-UPC/UCalib/blob/main/LICENSE). If you use UCalib in an academic work, please cite:

    @article{rs11232722,
      AUTHOR = {Simarro, Gonzalo and Calvete, Daniel and Souto, Paola},
      TITLE = {UCalib: Cameras autocalibration on coastal videomonitoring systems},
      JOURNAL = {Remote Sensing},
      VOLUME = {},
      YEAR = {2021},
      NUMBER = {},
      ARTICLE-NUMBER = {},
      URL = {},
      ISSN = {},
      DOI = {},
      NOTE = {Submitted}
      }
