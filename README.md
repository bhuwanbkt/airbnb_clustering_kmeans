# Airbnb Clustering and Analysis Pipeline

This project is part of the Data Mining course and implements a complete clustering pipeline for Airbnb listings using both traditional **K-Means** and **Bisecting K-Means** algorithms. It covers data preprocessing, feature engineering, clustering, visualization, and cluster analysis. 

---

## Features

- **Data Cleaning**: Handles missing values, removes outliers, converts date columns, and filters invalid rows.
- **Feature Engineering**: 
  - Log transformation for skewed numeric features.
  - Outlier capping (IQR, Z-score, or percentile-based).
  - Encoding categorical features using OneHotEncoding.
- **Clustering Algorithms**:
  - `simple_kmeans`: Classic implementation of K-Means.
  - `bisecting_kmeans`: Hierarchical K-Means approach for improved clustering.
- **Elbow Method**: Automatically determines the optimal number of clusters using inertia and `KneeLocator`.
- **Visualization**: Uses PCA for 2D cluster plots.
- **Cluster Analysis**: Prints detailed statistics and reviews activity for each cluster.

---

## Structure

- `load_and_clean_data(filepath)`: Loads and preprocesses Airbnb dataset.
- `cap_outliers(df)`: Removes extreme values from numerical features.
- `log_transform(df, columns)`: Applies `log1p` transformation for skewed distributions.
- `encoder_categorical(df, columns)`: Encodes categorical variables.
- `clustering_pipeline(df, cluster_cols, method)`: Runs complete clustering and visualization pipeline.
- `elbow_plot(X_scaled, method)`: Computes inertia scores for multiple values of `k`.
- `plot_clusters_pca(X_scaled, labels)`: Visualizes cluster separability using PCA.
- `cluster_analysis(df, labels)`: Summarizes key statistics and dominant traits per cluster.

---

## Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `kneed`
- `scipy`

You can install all dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed scipy
```

## Usage

Place your Airbnb data in a CSV file named airbnb.csv in the same directory
Run the script:

```bash
python airBNB_Clustering_kmean_Group_Backbencher.py
```
## Conclusion

The project successfully demonstrates the application of clustering to real-world data. The custom K-Means implementation performed well and aligned with `sklearn`'s output. Visualizations and evaluations confirmed the appropriateness of the selected number of clusters.

The detail explanation and results can be found in the pdf file `airBNB_Clustering_kmean_Group_Backbencher.pdf`

