
# UCalib

`UCalib` is an open source software written in Python for automatic image calibration of camera orientation from a set of manually calibrated images.

### Description
The calibration algorithm assumes that the camera position and the intrinsic parameters of the camera remains unchanged. The result of the process is a common position and intrinsic camera parameters for all images, and the orientation of the cameras for each of the images. The development of this software is suitable for Argus-type video monitoring stations. Details about the algorithm and methodology are described in
> *Simarro, G.; Calvete, D.; Luque, P.; Orfila, A.; Ribas, F. UBathy: A New Approach for Bathymetric Inversion from Video Imagery. Remote Sens. 2019, 11, 2722. https://doi.org/10.3390/rs11232722*

The automatic calibration process consists of two steps:
 1. [Manual calibration of the base](#base-calibration)
 2. [Automatic image calibration](#image-calibration)

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
  * `basis.py`
  * `images.py`
  * `ulises.py`

The local modules of `UCalib` are located in the **`ucalib`** folder:
- `basis.py`: Contains the main function for manual base calibration. 
- `images.py`: Contains the main function for automatic image calibration of the base. 
- `ulyses.py`: Contains core functions.

To run a demo with the images in folder **`basis`** and **`imagesToAutoCalibrate`** using a Jupyter Notebook we provide the file `example_notebook.ipynb`. For experienced users, the `example.py` file can be run in a terminal. 

## Base calibration
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
from ucalib import basis, images
```

Set the folder path where files for calibrating the base are located:


```python
pathFolderBasis = 'basis'
```

Run the calibration algorithm of the basis:


```python
basis.nonlinearCalibrationOfBasis(pathFolderBasis)
```

As a result of the calibration, the calibration file `<basisImage>cal.txt` is generated in the **`basis`** directory for each of the images. This file contains the following parameters:

| Magnitudes | Variables | Units |
|:--|:--:|:--:|
| Camera position coordinates | `xc`, `yc`, `zc` | _m_ |
| Camera orientation angles | `ph`, `sg`, `ta` | _rad_ |
| Lens radial distortion (quadratic, parabolic) | `k1a`, `k2a` | _-_ |
| Lens tangential distortion (quadratic, parabolic) | `p1a`, `p2a` | _-_ |
| Pixel size | `sc`, `sr` | _-_ |
| Decentering | `oc`, `rr` | _pixel_ |
| Image size | `nc`, `nr` | _pixel_ |
| Calibration error | `errorT`| _pixel_ |

The different calibration files `*cal.txt` differ only in the angles of the camera orientation  (`ph`, `sg`, `ta`) and the calibration error (`errorT`).

***info mesages error***_GONZALO_

## Image calibration

In this second step, each of the images in the folder **`imagesToAutoCalibrate`** will be automatically calibrated. Set the folder path where images to calibrate automatically are stored:


```python
pathFolderImagesToAutoCalibrate = 'imagesToAutoCalibrate'
```

Set the values of the automatic image calibration parameters:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Critical homography error | `fC` | _5._ | _pixel_ |
| Critical number of pairs | `KC` | _4_ | _-_ |
| Number of features to identify with ORB | `nORB` | _10000_ | _-_ |



```python
fC, KC = 5., 4
nORB = 10000
```

Run the algorithm to calibrate images automatically:


```python
images.autoCalibration(pathFolderBasis,pathFolderImagesToAutoCalibrate,fC,KC,nORB)
```

For each of the images `<image>.png` in directory **`imagesToAutoCalibrate`**, a calibration file `<image>cal.txt` with the same characteristics as the one described above will be obtained.
The self-calibration process may fail because the homography error is higher than the one set by the parameter fC, the number of pairs is lower than the critical value KC or ORF cannot identify pairs in the image. In any of these cases it is reported that the image  `<image>.png` is _not automatically calibratable_.

## Contact us

Are you experiencing problems? Do you want to give us a comment? Do you need to get in touch with us? Please contact us! To do so, we ask you to use the [Issues section](https://github.com/Ulises-ICM-UPC/UCalib/issues) instead of emailing us.

## License

UCalib is released under a [GPLv3 license](https://github.com/Ulises-ICM-UPC/UCalib/blob/main/LICENSE). If you use UCalib in an academic work, please cite:

    @article{rs11232722,
      AUTHOR = {Simarro, Gonzalo and Calvete, Daniel and Luque, Pau and Orfila, Alejandro and Ribas, Francesca},
      TITLE = {UBathy: A New Approach for Bathymetric Inversion from Video Imagery},
      JOURNAL = {Remote Sensing},
      VOLUME = {11},
      YEAR = {2019},
      NUMBER = {23},
      ARTICLE-NUMBER = {2722},
      URL = {https://www.mdpi.com/2072-4292/11/23/2722},
      ISSN = {2072-4292},
      DOI = {10.3390/rs11232722}
      }
  
