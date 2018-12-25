# Comparative Visualization of  Document Clustering Models

Models in comparison : LDA(Topic Modeling), K-Means, Deep Embedding Clustering

A visualization tool to compara document clustering results of clustering models.

Main Page: ![main_page](./resource/main_view.png)

Demo Video: ![](./resource/video_clip2.gif)


For Dev env

1. Need to install all required packages by
   `pip install -r ./requirements.txt`
2. Copy Spacy Data File
    * Download data file: `python -m spacy download en`
    * Copy data file("<Spacy Data Path>/en_core_web_sm-2.0.0") to "npl_data/en/en_core_web_sm-2.0.0" folder
3. Test local server by running
    `python3 py_source/run_app.py`
4. access web via browser by 'localhost:5000'

Packaging for Windows 64bit

1. PyInstaller
    * Package python codes by PyInstaller: `pyinstaller -Fw --distpath ./ ./packaging_task/run_app.spec`
