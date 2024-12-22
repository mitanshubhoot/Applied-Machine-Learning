Welcome to the **Repository**! 

This repository contains 8 projects, each focusing on a unique machine learning task. These projects explore data analysis, regression, classification, dimensionality reduction, clustering, and deep learning using diverse datasets. Below is a detailed description of each project and its objectives.

---

## **Project 1&2: Prediction**

### Objective:
Build a regression model to predict attributes such as GDP per capita, social support, freedom, and other socioeconomic indicators.

### Key Tasks:
1. **Data Analysis:**
   - Summarize the dataset (size, feature types).
   - Visualize distributions and analyze correlations.
2. **Feature Engineering:**
   - Compute statistical values and identify features requiring special treatment.
3. **Model Training:**
   - Train and evaluate linear regression models using closed-form solutions and SGD.
   - Explore Ridge, Lasso, and Elastic Net regularization.
   - Compare performance with polynomial regression.
4. **Evaluation:**
   - Test the model using a separate test set and evaluate its performance.

### Tools:
- Jupyter Notebook for analysis and visualization.
- Libraries: NumPy, Pandas, Matplotlib, Scikit-learn.

---

## **Project 3&4: Rock Classification**

### Objective:
Classify rock types (Igneous, Metamorphic, Sedimentary) using a dataset with 11 features.

### Key Tasks:
1. **Data Preparation:**
   - Process and split the dataset into training, validation, and testing sets.
   - Visualize feature distributions and analyze correlations.
2. **Model Training:**
   - Train classifiers such as Logistic Regression, SVM, and Random Forest.
   - Tune hyperparameters using grid search or manual exploration.
3. **Ensemble Methods:**
   - Combine individual classifiers into an ensemble for improved performance.
4. **Comparison with Human Performance:**
   - Compare model accuracy with human accuracy and analyze the results.

### Tools:
- Jupyter Notebook for analysis and training.
- Libraries: Scikit-learn, Matplotlib, Seaborn.

---

## **Project 5&6: PCA and Clustering on Rock Images**

### Objective:
Apply PCA and other dimensionality reduction techniques to analyze rock images and identify clusters.

### Key Tasks:
1. **PCA Analysis:**
   - Determine the number of components required to retain 90% variance.
   - Reconstruct and visualize images with reduced dimensionality.
2. **Visualization:**
   - Use PCA, t-SNE, LLE, and MDS to create 2D scatter plots of image embeddings.
3. **Comparison with Human Features:**
   - Use Procrustes analysis to compare embeddings with human-labeled features.
4. **Clustering:**
   - Perform clustering using K-Means and EM.
   - Visualize and evaluate clustering performance.
5. **Neural Networks:**
   - Build a feedforward neural network (using dense and CNN layers).
   - Procrustes analysis with human data.
### Tools:
- Image processing: OpenCV, Matplotlib.
- Dimensionality Reduction: Scikit-learn.
- Procrustes Analysis: SciPy.

---

## **Project 7&8: Working with Large Language Models**

### Objective:
Evaluate the performance of a pre-trained models on dataset.

### Key Tasks:
1. **Model Description:**
   - Analyze the architecture and parameter breakdown of the selected CLIP model.
2. **Image Classification:**
   - Evaluate model accuracy across five conditions (realistic, geons, silhouettes, blurred, features).
   - Compare results with human performance.
3. **Dimensionality Reduction and Visualization:**
   - Use t-SNE to visualize embeddings in 2D space.
   - Add object images to the embedding visualization.
4. **Analysis:**
   - Compare model embeddings with human-labeled features using Procrustes analysis.
   - Compute correlation coefficients and analyze significance.

### Tools:
- Pre-trained models: Huggingface Transformers.
- Visualization: Matplotlib, t-SNE.
- Libraries: NumPy, SciPy, Scikit-learn.
