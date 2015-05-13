# Cascade 分类器模型（供 OpenCV 使用的模型）

[README in Chinese 中文版本请点此](README.zh_CN.md)

使用 opencv_traincascade 训练出来的模型。部分模型使用 OpenCV 2.4 的 opencv_traincascade 训练，部分模型使用 OpenCV 3.0 的 opencv_traincascade 训练。

根据目前的测试情况，haarcascade764 模型是表现最好的。


## 模型说明

### haarcascade1463

使用了 1463 个正样本，haar-like 特征。

### haarcascade764

使用了 764 个正样本，haar-like 特征。

### haarcascade764-bigneg

使用了 764 个正样本和几千个负样本，haar-like 特征。

### haarcascade764hog

使用了 764 个正样本，HOG 特征。

### haarcascade764lbp

使用了 764 个正样本，LBP 特征。

### haarcascade-853-7708-haar

使用了 853 个正样本，7708 个负样本，haar-like 特征。正样本仅包含车辆前视图和后视图，不包含侧视图。

