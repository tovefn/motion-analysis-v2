The structure of this project is a bit convoluted, sorry about that..

This repo can be seen as the top level, mainly containing code to generate datasets. Running install-repos script as bellow will clone repos for keypoint extraction in videos ([mmpose](https://github.com/filipkro/mmpose.git) and [mmdet](https://github.com/filipkro/mmdetection.git)) into `pose/` as well as code to train and validate models for classification of the motions (from https://github.com/filipkro/tsc.git).

Both keypoint extraction and model training can probably be run locally, but will take some time unless you have some compute (preferably with GPUs)... During my MSc thesis I had access to the [Alvis cluster](https://www.c3se.chalmers.se/about/Alvis/).

## Overview
For running the keypoint extraction [run-detection](run-detection) adn [run-detection-folder](run-detection-folder) are example sbatch files running the extraction on a single video or a folder on Alvis (note, they worked 3 years ago, I don't know what has changed now etc, I didn't use containers for this unfortunately...).

Running on a folder will call [`pose/analysis/run-folder-cluster.sh`](pose/analysis/run-folder-cluster.sh) which calls [`analyse_folder.py`](pose/analysis/analyse_folder.py) in the same directory. This will in turn run keypoint extraction ([`analyse_vid.py`](pose/analysis/analyse_vid.py)) on all videos in the specified directory (how this path is spoecified and resolved might have to be changed depending on where and how it is run and organised). When extracting keypoints the frames are flipped such that all exercises are "conducted" on the right hand side leg. Some videos had this information in the file name, others hadn't. When it couldn't be identified from the filename the keypoint positions were analysed - how this is done needs top be modified for other motions.

In `pose/analysis/utils` there are a bunch of `create_POE_*.py` files. These are used to create the actual dataset used for specific POEs. For the different POEs different keypoints are used etc. These are atm incredibly messy. Note that for train and test splits repetitions from the same individual should not be in both datasets. Finding which keypoints to use for different POEs wasn't trivial, a lot of time can be spent on this. I looked at gradients for different channels to try to identify important features (more info on this in the classification code, and I wrote a bit about in the thesis).


# motion-analysis

master's thesis project, assessments of POEs in videos.
report: [Visual assessments of Postural Orientation Errors using ensembles of Deep Neural Networks](https://github.com/filipkro/motion-analysis/blob/master/tex/mt-motion-analysis.pdf)

## install repos:
```
$ cd motion-analysis
$ ./scripts/install-repos.sh
```

I ran with Python Python 3.7

install dependencies for mmpose (preferably from within some virtual env):
```
$ ./install/install.sh
```

## Running some of the code
To run the code as described below dependencies as well as `mmpose` and `mmdet` needs to be installed as described above.

To analyse videos in one folder the following command can be run (from `pose/analysis`)
```
python analyse_folder.py pose/mmpose/configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py pose/mmpose/checkpoints/top-down/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth <path to video directory> --out-video-root <save path> --folder_box pose/mmpose/mmdetection/ --show true --save_pixels false
```
`pose/mmpose/configs/top_down/darkpose/coco/hrnet_w48_coco_384x288_dark.py` dscribes the model extracting key points, `pose/mmpose/checkpoints/top-down/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth` contains the weights for this model, then there are a bunch of extra flags which can be provided, eg, setting `--show` as `true` will show each frame with keypoints during extraction, `save_pixels` sets whether the pixel position or a normalised position of the keypoints is saved.

To create the actual datasets (ie going from multiple numpy arrays of all keypoints describing all repetions of single videos to one array per POE of shape (`total nbr of repetitions` x `time series length` x `nbr of features`) the following command can be run (from `pose/analysis/utils`)
```
python create_POE_consensus.py <path to directory with analysed videos and resulting keypoints> --rate 25 --debug f --info_file t --save_path <directory to save dataset in>
```
This will go through all `.npy` files in the specified directory, resample them to the frame rate specified by `rate`, identify the repetition splits, and then extract features which are specified in the code (e.g. x-position of the right hand side knee, difference between y-position of right and left hip, angle between right hand side hip and knee). Which POE data is being created for is also specified in the code - would make sense to set as an argument instead. Note that there are different versions of this script `create_POE_consensus.py` is the most recent, running on consensus data. Primarily consider this version or `create_POE_alldata.py`.

For the classification I used ensembles of a few different models, some used datasets with a normalised length (all subjects will perform the motions at different pace - this was an attempt to reduce the importance of this factor). To do this the following can be run (from `pose/analysis/utils`)
```
python same-len.py <path to data>/data_foot_train.npz
```
This will normalise it to length 100, change `data_foot_train.npz` to whatever dataset file you want to do this for.
