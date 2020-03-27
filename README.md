Mask R-CNN applied on Cloth segmentation

Used Fair Detectron2

Data: DeepFashion2

Example of the results:

![alt text](https://github.com/PLXIV/cloth_segmentator/blob/master/example/001099_8.jpg)


json:

{"colors": [["#080606", 0.9417353427734523], ["#635751", 0.035043785760900906], ["#ead9b9", 0.023220871465646768]], "class_predicted": 8, "score": 0.9914104342460632, "image_path": "segmented_images/images/001099_8.jpg"}


| Perfectly classified:                   | 0.5904270208067677  |
|-----------------------------------------|---------------------|
| Almost well classified:                 | 0.03181662675333562 |
| Partially classified:                   | 0.2703635741610425  |
| More predictions than required (ratio): | 0.3903140457839641  |
| Missing predictions (ratio):            | 1.2020016104912     |
| Errors :                                | 0.10739277827885423 |
| Nothing predicted:                      | 0.0                 |
| Wrongly classified:                     | 0.10739277827885423 |


Taking into account that Deepfashion2 achieved 0.66 overall accuracy on the same dataset and that I just trained during 20h. on a gtx1070, the results are not bad.

It really needs to be refactored, but I do not have time for it now :)
