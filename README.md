# Learning for sensor data. Common approaches and a use-case application.

This repository contains the code written for conducting the studies of common approaches in learning from sensor data. This study was done following the curriculum of the master's programme in Data Science at the Chemnitz University of Technology. Urgrow became a base for the study.

## Structure

The study consists of two sub-studies. The first one aims to improve forecasting results of sequential time series. For that purpose, three deep learning models were compared on different datasets. The DILATE loss function implemented in PyTorch was used in one of the models and showed its superiority over two other models on long non-stationary sequential data. However, there was no significant difference on a stationary dataset.

The second sub-study aims to find an efficient, yet fast algorithm for time-series clustering. Two partitional and one hierarchical algorithms were compared by their Rand statistics and time. Hierarchical agglomerative clustering outperformed the k-means and k-shape on synthetic temperature, air pressure and humidity datasets.  



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

"@incollection{leguen19dilate,
title = {Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models},
author = {Le Guen, Vincent and Thome, Nicolas},
booktitle = {Advances in Neural Information Processing Systems},
pages = {4191--4203},
year = {2019}
}"