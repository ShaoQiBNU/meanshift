meanshift算法
=================

# 一. meanshift算法简介

> meanshift算法是基于核密度估计的爬山算法，可用于聚类、图像分割、跟踪等，mean shift是一个向量，它的方向指向当前点上概率密度梯度的方向。核密度估计算法指的是根据数据概率密度不断移动其均值质心，直到满足一定条件。通俗地讲，meanshift算法可以看作是使多个随机中心点向着密度最大的方向移动，最终得到多个最大密度中心。可以看成初始有多个随机初始中心，每个中心都有一个半径为bandwidth的圆，我们要做的就是求解一个向量，使得圆心一直往数据集密度最大的方向移动，也就是每次迭代的时候，都是找到圆里面点的平均位置作为新的圆心位置，直到满足某个条件不再迭代，这时候的圆心也就是密度中心。 如图1所示：
![image](https://github.com/ShaoQiBNU/meanshift/blob/master/images/1.png)


> 多维数据集聚类过程如下：
> 1. 在未被标记的数据点中随机选择一个点作为中心center；
> 2. 以center为圆心，bandwidth为半径做一个高维球，落在这个球内的所有点，记做集合M，认为这些点属于簇c。同时，把这些球内点属于这个类的概率加1，这个参数将用于最后步骤的分类
> 3. 以center为中心点，计算从center开始到集合M中每个元素的向量，将这些向量相加，得到向量shift。
> 4. center = center+shift。即center沿着shift的方向移动，移动距离是||shift||。
> 5. 重复步骤2、3、4，直到shift的大小很小（就是迭代到收敛），记住此时的center。注意，这个迭代过程中遇到的点都应该归类到簇c。
> 6. 如果收敛时当前簇c的center与其它已经存在的簇c2中心的距离小于阈值，那么把c2和c合并。否则，把c作为新的聚类，增加1类。
> 7. 重复1、2、3、4、5直到所有的点都被标记访问。
> 8. 分类：根据每个类，对每个点的访问频率，取访问频率最大的那个类，作为当前点集的所属类。 

# 二. 算法实现


```python
#coding:utf-8   
  
import numpy as np   
from sklearn.cluster import MeanShift, estimate_bandwidth   
from sklearn.datasets.samples_generator import make_blobs   
  
  
# 生成样本点   
centers = [[1, 1], [-1, -1], [1, -1]]   
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)   
  
  
# 通过下列代码可自动检测bandwidth值   
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)   
  
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)   
ms.fit(X)   
labels = ms.labels_   
cluster_centers = ms.cluster_centers_   
  
labels_unique = np.unique(labels)   
n_clusters_ = len(labels_unique)   
new_X = np.column_stack((X, labels))   
  
print("number of estimated clusters : %d" % n_clusters_)   
print("Top 10 samples:",new_X[:10])   
  
# 图像输出   
import matplotlib.pyplot as plt   
from itertools import cycle   
  
plt.figure(1)   
plt.clf()   
  
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')   
for k, col in zip(range(n_clusters_), colors):   
    my_members = labels == k   
    cluster_center = cluster_centers[k]   
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')   
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,   
             markeredgecolor='k', markersize=14)   
plt.title('Estimated number of clusters: %d' % n_clusters_)   
plt.show() 

```
