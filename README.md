# Scene Text Recognition by Reading Individual Characters

### Demo

https://colab.research.google.com/drive/1nz6SZgc6J_i5w6ZkUi1_ZrN7f3GusBy_#scrollTo=8pJTGXiycl3M

### Weights

- Classifier: https://drive.google.com/file/d/1rDbsqBFRpiby7FTfJAEZdiiAdgtmKMb1/view?usp=sharing (for alphabet of `datasets.ALPHANUMERIC`)

- Detector: https://drive.google.com/file/d/1_vD0pK2qAAobfe55C7wPbTOV4SJj2YEI/view?usp=sharing

### Algorithms and Models Used

### Results

The model was evaluated with the ICDAR2013 test dataset. Its website lists four
independent tasks with which any text recognition model can be evaluated. Only
tasks 1, 3, and 4 are relevant for our purposes.

The evaluation scripts for each task were provided by the ICDAR2013 website.

1. **Text Localization** - Given scene text images, predict word-level bounding
boxes for each image.
  - Recall: `61.21%`
  - Precision: `53.01%`
  - H-mean: `56.82%`

2. **Character Segmentation** - Given scene text images, predict the character
masks for each image.
  - Not Applicable

3. **Word Recognition** - Given images of words cropped from scene text images,
predict the word in the image.
  - Correctly Recognized Words (case insensitive): `36.89%`
  - Correctly Recognized Words (case sensitive): `20.55%`
  - Total Edit Distance (case insensitive): `478.41`
  - Total Edit Distance (case sensitive): `605.81`

4. **End to End Recognition** - Given scene text images, predict word bounding
boxes and corresponding word inside.
  - Recall: `19.96%`
  - Precision: `13.48%`
