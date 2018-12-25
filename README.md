# Comparative Visualization of  Document Clustering Models

Models in comparison : LDA(Topic Modeling), K-Means, Deep Embedding Clustering

A visualization tool to compara document clustering results of clustering models.



## Overview

* Juxtapose 3 Models that using different techniques to Topic Modeling
![main_page](./resource/main_view.png)


* Demo Video
![Demo_Video](./resource/video_low.gif)


## Project Description

Document clustering is one of the major tasks in natural language processing fields. While the widely researches for document clustering, it has several limitations because of the clustering is one of the unsupervised learning techniques. The most critical limitations are anyone cannot be guaranteed that the clustering result is right. This means that someone who is the expert having domain-specific knowledge has to check and verify the clustering result.

Due to this reason, Grace and I catch up the needs to compare clustering results from different clustering techniques and we visualized different clustering model on a web page to support the evaluation of models and quality of clusters to retrieve target clusters, which often becomes an exploratory problem.

---

## Clustering Algorithms

Clustering models are divided into several types according to using algorithms. We choose 3 different type models, Latent Dirichlet Allocation(LDA), K-means, Deep Embedded Clustering(DEC).
* Latent Dirichlet Allocation(_Porbabilistic methods_): LDA has considered each item is a mixture of various clusters with probabilistic distribution.
* K-Means(_Centroid-based methods_): K-Means decides the cluster of each item to minimize the within-cluster sum of squares(WCSS, variance)
* Deep Embedded Clustering(_Low dimensional embedding method_): DEC uses a deep neural network to choose features to represent each cluster

## Dimension Reduction

![U-MAP vs t-SNE](https://user-images.githubusercontent.com/8471958/50389355-a5bcff80-076c-11e9-9424-d8b2ff8fb9e3.png)

<center>U-MAP vs t-SNE<br><i>Umap: Uniform manifold approximation and projection for dimension reduction(McInnes, L., & Healy, J., 2018)</i></center>

To visualize documents, we used Uniform Manifold Approximation(U-MAP). t-SNE is the most popular technique to reduce dimensions for preprocessing and projection. However, using the t-SNE need for powerful computing power and high time-consuming. In recently, a new technique called U-MAP developed. This technique preserves more of the global structure and has a faster time complexity than t-SNE

## Extraction Keywords & Entity Recognition

![keywords_extraction](https://user-images.githubusercontent.com/8471958/50389570-372d7100-076f-11e9-8026-cd0ddce68d41.png)

<center>Extraction Keywords</center>

For verifying clusters, we extract main keywords from each cluster. Keywords are determined by probability based on the frequency of a word in each cluster over frequency in overall documents. We use the pyLDAvis library to extract keywords. However, pyLDAvis is not supported extracting keywords from only LDA model. Because of that, we use a custom library called [kmeans to pyLDAvis](https://github.com/lovit/kmeans_to_pyLDAvis) made by lovit to extract keywords from the K-means and DEC model.

![entity_recognition](https://user-images.githubusercontent.com/8471958/50390293-da36b880-0778-11e9-8442-a887e6344b23.png)

<center>Entity Recognition</center>

Entity Recognition is another information to verify the cluster quality. We provide not only entities of each document, but also the frequency of entities of documents in the same cluster. To recognize entities, we use the spaCy library with English dataset(en_core_web_sm, including Vocabulary, Syntax, Entities).

## Implementation

![overall_architecture](https://user-images.githubusercontent.com/8471958/50390304-fe929500-0778-11e9-8131-95e456ed0b2a.png)

<center>Overall architecture of application</center>

We use basically python to implement this system including running model, importing model, feature reduction(extracting coordinates for scatter chart) and entity recognition. Using Flask(python library), we make the local web server, and every result from python have represented this server. On the front side, we visualize data from Flask server using javascript and d3.js. 

Moreover, we make the application using Electron that is one of the famous platforms to build cross-platform desktop apps. Electron makes the window including chromium browser, therefore our application can call web pages from Flask server. For this, we package the python server by pyinstaller and the front web pages by Electron.

## Usage

* **For Dev environment**

1. Need to install all required packages by
   `pip install -r ./requirements.txt`
2. Copy Spacy Data File
    * Download data file: `python -m spacy download en`
    * Copy data file("\<Spacy Data Path>/en_core_web_sm-2.0.0") to "npl_data/en/en_core_web_sm-2.0.0" folder
3. Test local server by running
    `python3 py_source/run_app.py`
4. access web via browser by 'localhost:5000'

* **Packaging for Windows 64bit**

1. PyInstaller
    * Package python codes by PyInstaller: `pyinstaller -Fw --distpath ./ ./packaging_task/run_app.spec`
    * Run binary file: `./run_flask.exe` or `npm start`
