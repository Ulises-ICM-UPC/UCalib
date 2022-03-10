
# UCalib

`UCalib` is an open source software written in Python for automatic image calibration from a set of images that are manually calibrated.

### Description
The calibration algorithm assumes that the camera position and the intrinsic parameters of the camera remain unchanged. The result of the process is a common position and intrinsic camera parameters for all images, and the orientation of the cameras for each of the images. In addition, planviews can be generated for each image. The development of this software is suitable for Argus-type video monitoring stations. Details about the algorithm and methodology are described in
> *Simarro, G.; Calvete, D.; Souto, P. UCalib: Cameras Autocalibration on Coastal Video Monitoring Systems. Remote Sens. 2021, 13, 2795. https://doi.org/10.3390/rs13142795*

The automatic calibration process consists of two steps:

1. [Basis calibration](#basis-calibration)
2. [Automatic image calibration](#automatic-image-calibration)

Further UCalib allows to generate planviews for the calibrated images:

3. [Planview generation](#planviews)
 
A code to verify the quality of the GCPs used in the manual calibration of the basis images is also provided:

4. [Check GCP for basis calibration](#gcp-check)

### Requirements and project structure
To run the software it is necessary to have Python (3.8) and install the following dependencies:
- cv2 (4.2.0)
- numpy (1.19.5)
- scipy (1.6.0)

In parenthesis we indicate the version with which the software has been tested. It is possible that it works with older versions. 

The structure of the project is the following:
* `example.py`
* `example_notebook.py`
* **`ucalib`**
  * `ucalib.py`
  * `ulises_ucalib.py`
* **`example`**
  * **`basis`**
    * `basisImage01.png`
    * `basisImage01cal0.txt`
    * `basisImage01cal.txt`
    * `basisImage01cdg.txt`
    * `basisImage01cdh.txt`
    * . . .
  * **`basis_check`**
    * `basisImage01.png`
    * `basisImage01cdg.txt`
    * . . .
  * **`images`**
    * `image000001.png`
    * `image000001cal.txt`
    * . . .
  * **`planviews`**
    * `crxyz_planviews.txt`
    * `xy_planview.txt`
    * `image000001plw.png`
    * . . .
  * **`TMP`**
    * `basisImage01cal0_check.png`
    * `basisImage01cal_check.png`
    * `image000001cal_check.png`
    * `image000001_checkplw.png`
    * `image000001plw_check.png`
    * . . .

The local modules of `UCalib` are located in the **`ucalib`** folder.

To run the demo in the folder **`example`** with the basis of images in **`basis`** and the images in **`images`** using a Jupyter Notebook we provide the file `example_notebook.ipynb`. For experienced users, the `example.py` file can be run in a terminal. `UCalib` handles `PNG` (recommended) and `JPEG` image formats.

## Basis calibration
To manually calibrate the images selected for the basis, placed in the folder **`basis`**,  it is necessary that each image `<basisImage>.png` is supplied with a file containing the Ground Control Points (GCP) and, optionally, the Horizon Points (HP). The structure of each of these files is the following:
* `<basisImage>cdg.txt`: For each GCP one line with  (minimum 6)
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`
* `<basisImage>cdh.txt`: For each HP one line with (minimum 3)
>`pixel-column`, `pixel-row`

Quantities must be separated by at least one blank space between them and the last record should not be continued with a newline (return).

To generate `<basisImage>cdg.txt` and `<basisImage>cdh.txt` files the [UClick](https://github.com/Ulises-ICM-UPC/UClick) software is available.

### Run basis calibration
Import modules:


```python
import sys
import os
sys.path.insert(0, 'ucalib')
import ucalib as ucalib
```

Set the main path and the path where the basis is located:


```python
pathFolderMain = 'example'
pathFolderBasis = pathFolderMain + os.sep + 'basis'
```

Set the value of maximum error allowed for the basis calibration:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Critical reprojection pixel error | `eCritical` | _5._ | _pixel_ |



```python
eCritical = 5.
```

Select an intrinsic camera calibration model.

| Camara model | `parabolic` | `quartic`| `full` |
|:--|:--:|:--:|:--:|
| Lens radial distortion | parabolic | parabolic + quartic | parabolic + quartic |
| Lens tangential distortion | no | no | yes |
| Square pixels | yes | yes | no |
| Decentering | no | no | yes |

The `parabolic` model is recommended by default, unless the images are highly distorted.


```python
calibrationModel = 'parabolic'
```

To facilitate the verification that the GCPs have been correctly selected in each image of the basis, images showing the GCPs and HPs (black), the reprojection of GCPs (yellow) and the horizon line (yellow) on the images can be generated. Set parameter `verbosePlot = True`, and to `False` otherwise. Images (`<basisImage>cal0_check.png`) will be placed on a TMP folder.


```python
verbosePlot = True
```

Run the initial calibration algorithm for each image of the basis:


```python
ucalib.CalibrationOfBasisImages(pathFolderBasis, eCritical, calibrationModel, verbosePlot)
```

In case that the reprojection error of a GCP is higher than the error `eCritical` for a certain image `<basisImage>`, a message will appear suggesting to re-run the calibration of the basis or to modify the values or to delete points in the file `<basisImage>cdg.txt`. If the calibration error of an image exceeds the error `eCritical` the calibration is given as _failed_. Consider re-run the calibration of the basis or verify the GPCs and HPs.

Then, run the algorithm to obtain the position and the optimal intrinsic parameters of the camera:


```python
ucalib.CalibrationOfBasisImagesConstantXYZAndIntrinsic(pathFolderBasis, calibrationModel, verbosePlot)
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

The different calibration files `<basisImage>cal.txt` differ only in the angles of the camera orientation  (`ph`, `sg`, `ta`) and the calibration error (`errorT`). A `<basisImage>cal0.txt` file with the initial calibration parameters for each image of the basis will also have been generated.

## Automatic image calibration

In this second step, each of the images in the folder **`images`** will be automatically calibrated. Set the folder path where images to calibrate automatically are stored. To facilitate the verification of the calibration of each image, images showing the reprojection of the GCPs and the horizon line can be generated. Set parameter `verbosePlot = True`, and to `False` otherwise. Images(`<images>cal_check.png`) will be placed on a **`TMP`** folder.


```python
pathFolderImages = pathFolderMain + os.sep + 'images'
verbosePlot = True
```

Set the values of the automatic image calibration parameters:

|  | Parameter | Suggested value | Units |
|:--|:--:|:--:|:--:|
| Number of features to identify with ORB | `nORB` | _10000_ | _-_ |
| Critical homography error | `fC` | _5._ | _pixel_ |
| Critical number of pairs | `KC` | _4_ | _-_ |



```python
nORB, fC, KC = 10000, 5., 4
```

Run the algorithm to calibrate images automatically:


```python
ucalib.AutoCalibrationOfImages(pathFolderBasis, pathFolderImages, nORB, fC, KC, verbosePlot)
```

For each of the images `<image>.png` in directory **`images`**, a calibration file `<image>cal.txt` with the same characteristics as the one described above will be obtained. The autocalibration process may fail because the homography error is higher than the one set by the parameter `fC`, the number of pairs is lower than the critical value `KC` or `ORB` not being able to identify pairs in the image. In any of these cases it is reported that the calibration of the image  `<image>.png` has _failed_.

## Planviews

Once the frames have been calibrated, planviews can be generated. The region of the planview is the one delimited by the minimum area rectangle containing the points of the plane specified in the file `xy_planview.txt` in the folder **`planviews`**. The planview image will be oriented so that the nearest corner to the point of the first of the file  `xy_planview.txt` will be placed in the upper left corner of the image. The structure of this file is the following:
* `xy_planview.txt`: For each points one line with 
> `x-coordinate`, `y-coordinate`

A minimum number of three not aligned points is required. These points are to be given in the same coordinate system as the GCPs.

Set the folder path where the file `xy_planview.txt` is located and the value of `z0`.


```python
pathFolderPlanviews = pathFolderMain + os.sep + 'planviews'
z0 = 3.2
```

The resolution of the planviews is fixed by the pixels-per-meter established in the parameter `ppm`. To help verifying that the points for setting the planview are correctly placed, it is possible to show such points on the frames and on the planviews. Set the parameter `verbosePlot = True`, and to `False` otherwise. The images (`<image>_checkplw.png` and `<image>plw_check.png`) will be placed in a TMP folder.


```python
ppm = 2.0
verbosePlot = True
```

Run the algorithm to generate the planviews:


```python
ucalib.PlanviewsFromImages(pathFolderImages, pathFolderPlanviews, z0, ppm, verbosePlot)
```

As a result, for each of the calibrated images `<image>.png` in folder **`images`**, a planview `<image>plw.png` will be placed in the folder **`planviews`**. Note that objects outside the plane at height `z0` will show apparent displacements due to real camera movement. In the same folder, the file `crxyz_planview.txt` will be located, containing the coordinates of the corner of the planviews images:
* `crxyz_planview.txt`: For each corner one line with 
>`pixel-column`, `pixel-row`, `x-coordinate`, `y-coordinate`, `z-coordinate`

## GCP check

To verify the quality of the GCPs used in the manual calibration of the basis images, a RANSAC (RANdom SAmple Consensus) is performed. Points of the files `<basisImage>cdg.txt` located at the **`basis_check`** folder will be tested. The calibration of the points (minimum 6) is done assuming a _parabolic_ camera model and requires a minimum error `eCritical`. Set the folder and run the RANSAC algorithm:


```python
pathFolderBasisCheck = pathFolderMain + os.sep + 'basis_check'
ucalib.CheckGCPs(pathFolderBasisCheck, eCritical)
```

For each file `<basisImage>cdg.txt`, the GCPs that should be revised or excluded will be reported.

## Contact us

Are you experiencing problems? Do you want to give us a comment? Do you need to get in touch with us? Please contact us!

To do so, we ask you to use the [Issues section](https://github.com/Ulises-ICM-UPC/UCalib/issues) instead of emailing us.

## Contributions

Contributions to this project are welcome. To do a clean pull request, please follow these [guidelines](https://github.com/MarcDiethelm/contributing/blob/master/README.md).

## License

UCalib is released under a [GPLv3 license](https://github.com/Ulises-ICM-UPC/UCalib/blob/main/LICENSE). If you use UCalib in an academic work, please cite:

    @Article{rs13142795,
      AUTHOR = {Simarro, Gonzalo and Calvete, Daniel and Souto, Paola},
      TITLE = {UCalib: Cameras Autocalibration on Coastal Video Monitoring Systems},
      JOURNAL = {Remote Sensing},
      VOLUME = {13},
      YEAR = {2021},
      NUMBER = {14},
      ARTICLE-NUMBER = {2795},
      URL = {https://www.mdpi.com/2072-4292/13/14/2795},
      ISSN = {2072-4292},
      DOI = {10.3390/rs13142795}
      }

    @Online{ulisesdrone, 
      author = {Simarro, Gonzalo and Calvete, Daniel},
      title = {UCalib},
      year = 2021,
      url = {https://github.com/Ulises-ICM-UPC/UCalib}
      }
