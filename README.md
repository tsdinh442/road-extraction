# road-extraction
🌍 Enhancing Urban Planning and Development with Satellite Imagery🛰️

Understanding a site's infrastructure and connectivity is vital for urban planners and land developers. Yet, accessing comprehensive road data remains a challenge, especially in underdeveloped regions.

This project aims to uncover the potential of extracting road surfaces from high-resolution satellite imagery. Using the UNet architecture and training on Deep Glove datasets resized to 256 by 256 pixels, the model showcases impressive capabilities in identifying road information, even with the reduced pixel dimensions. 

## Required packages
```
  numpy
  tensorflow
  keras
  opencv-python
  scikit-learn
  matplotlib
```

## Dataset
The dataset was downloaded from Kaggle and contains 6226 satellite imagery in RGB, size 1024x1024.
```
@InProceedings{DeepGlobe18,
 author = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
 title = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
 booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 month = {June},
 year = {2018}
}
```

```
#!pip install kaggle
#!kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset datasets
```
## Result
![Example Image](result.png)
