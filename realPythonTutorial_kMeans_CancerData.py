## from real python tutorial
## kmeans pipeline with Cancer Data 

import tarfile
import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

uci_tcga_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
archive_name = "TCGA-PANCAN-HiSeq-801x20531.tar.gz"

# Build the url
full_download_url = urllib.parse.urljoin(uci_tcga_url, archive_name)

# Download the file
r = urllib.request.urlretrieve (full_download_url, archive_name)

# Extract the data from the archive
tar = tarfile.open(archive_name, "r:gz")
tar.extractall()
tar.close()

datafile = "TCGA-PANCAN-HiSeq-801x20531/data.csv"
labels_file = "TCGA-PANCAN-HiSeq-801x20531/labels.csv"

## convert to numpy array for scikit-learn
data = np.genfromtxt(
    datafile,
    delimiter=",",
    usecols=range(1, 20532),
    skip_header=1
)

true_label_names = np.genfromtxt(
    labels_file,
    delimiter=",",
    usecols=(1,),
    skip_header=1,
    dtype="str"
)

print(data[:5, :3])

print('true labels names: ', true_label_names[:5])

## To use these labels in the evaluation methods, you first need to convert the abbreviations to integers with LabelEncoder:
label_encoder = LabelEncoder()

true_labels = label_encoder.fit_transform(true_label_names)

print('true labels encoded: ' , true_labels[:5])

print(label_encoder.classes_)

n_clusters = len(label_encoder.classes_)

## pre-processing 
## 1. feature scaling
## 2. dimensionality reduction 
## Principal Component Analysis (PCA) is one of many dimensionality reduction techniques. 
# PCA transforms the input data by projecting it into a lower number of dimensions called components. 
# The components capture the variability of the input data through a linear combination of the input data’s features. 

## do both steps with the sk-learn pipeline class 
## here we use minmax to feature scale and pca to dim reduction 
preprocessor = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("pca", PCA(n_components=2, random_state=42)),
    ]
)

## Build the k-means clustering pipeline with user-defined arguments in the KMeans constructor:
clusterer = Pipeline(
   [
       (
           "kmeans",
           KMeans(
               n_clusters=n_clusters,
               init="k-means++",
               n_init=50,
               max_iter=500,
               random_state=42,
           ),
       ),
   ]
)

## The Pipeline class can be chained to form a larger pipeline:
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("clusterer", clusterer)
    ]
)

## execute:
pipe.fit(data)

## Evaluate the performance by calculating the silhouette coefficient:
preprocessed_data = pipe["preprocessor"].transform(data)

predicted_labels = pipe["clusterer"]["kmeans"].labels_

print('silhouette score: ' , silhouette_score(preprocessed_data, predicted_labels))

## Calculate ARI, too, since the ground truth cluster labels are available:
print('ARI = ', adjusted_rand_score(true_labels, predicted_labels))

## n_components=2 in the PCA step of the k-means clustering pipeline, you can also visualize the data in the context of the true labels and predicted labels
pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["component_1", "component_2"],
)

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    style="true_label",
    palette="Set2",
)

scat.set_title(
    "Clustering results from TCGA Pan-Cancer\nGene Expression Data"
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()

## The Pipeline class is powerful when you want to search the parameter space. 
# It allows you to perform basic parameter tuning using a for loop. 
# Empty lists to hold evaluation metrics
silhouette_scores = []
ari_scores = []
for n in range(2, 11):
    # This set the number of components for pca,
    # but leaves other steps unchanged
    pipe["preprocessor"]["pca"].n_components = n
    pipe.fit(data)

    silhouette_coef = silhouette_score(
        pipe["preprocessor"].transform(data),
        pipe["clusterer"]["kmeans"].labels_,
    )
    ari = adjusted_rand_score(
        true_labels,
        pipe["clusterer"]["kmeans"].labels_,
    )

    # Add metrics to their lists
    silhouette_scores.append(silhouette_coef)
    ari_scores.append(ari)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(6, 6))
plt.plot(
    range(2, 11),
    silhouette_scores,
    c="#008fd5",
    label="Silhouette Coefficient",
)
plt.plot(range(2, 11), ari_scores, c="#fc4f30", label="ARI")

plt.xlabel("n_components")
plt.legend()
plt.title("Clustering Performance as a Function of n_components")
plt.tight_layout()
plt.show()

## There are two takeaways from this figure:

##    The silhouette coefficient decreases linearly. The silhouette coefficient depends on the distance between points, 
# so as the number of dimensions increases, the sparsity increases.

##    The ARI improves significantly as you add components. It appears to start tapering off after n_components=7, 
# so that would be the value to use for presenting the best clustering results from this pipeline.
