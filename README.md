# D2ACE
D2ACE: Multi-Label Batch Selection Guided by Dual Dynamics and Adaptive Correlation Enhancement

# Package:
Python==3.12.4 numpy==1.26.4 scikit-multilearn==0.2.0 torch==2.4.1

# Usage:
## 数据集文件夹
Multi-label datasets can be downloaded from https://mulan.sourceforge.net/datasets-mlc.html.
Create a data folder in the current directory, and place the downloaded dataset files into it.

## run main.py
In detail, we can train on different datasets:

**Dataset used**

```python
path_to_arff_files = ["cal500", "birds", "enron", "scene", "yeast", "Corel5k", "rcv1subset1", "rcv1subset2", "rcv1subset3", "bibtex", "yahoo-Arts1", "yahoo-Business1", "mediamill"]
```

 **Label count of corresponding dataset:**

```python
label_counts = [174, 19, 53, 6, 14, 374, 101, 101, 101, 159, 25, 28, 101]
```

**Feature retention ratio of the corresponding dataset:**

```python
select_feature = [1, 1, 1, 1, 1, 1, 0.02, 0.02, 0.02, 1, 0.05, 0.05, 1]
```

# Key code
