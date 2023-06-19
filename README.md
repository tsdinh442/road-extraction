# Road Extraction from Satellite Images
🌍 Enhancing Urban Planning and Development with Satellite Imagery🛰️

Understanding a site's infrastructure and street connectivity is crucial for urban planners and land developers. Yet, accessing comprehensive road data remains a challenge, especially in underdeveloped regions.

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
The **DeepGlobe 2018 Dataset** is a collection of satellite images designed for the DeepGlobe Challenge. It consists of high-resolution satellite imagery covering various regions of the Earth. The dataset is intended for tasks such as semantic segmentation and object detection in satellite images.

![Samples](samples.png)


```
!pip install kaggle
!kaggle datasets download -d balraj98/deepglobe-road-extraction-dataset datasets
```
citation

```
@InProceedings{DeepGlobe18,
 author = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
 title = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
 booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 month = {June},
 year = {2018}
}

```

## Result
![Results](result.png)
