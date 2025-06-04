# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from scipy.stats import shapiro,skew

# Global Variables
LOG_COLS = ['price', 'minimum_nights', 'number_of_reviews','days_since_last_review', 'availability_365','reviews_per_month','calculated_host_listings_count']
NUMERIC_COLS = ['price', 'price_per_night', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                'longitude', 'latitude', 'days_since_last_review','calculated_host_listings_count']
GEO_COLS = ['longitude', 'latitude']
CATEGORICAL_COLS = ['room_type', 'neighbourhood_group']
AVAILABILITY = ['availability_365']

# Data Loading and Cleaning
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(['id', 'host_id', 'host_name', 'name'], axis=1)
    df = df[df['price'] > 0]
    df = df[df['minimum_nights'] > 0]
    df = df[df['minimum_nights'] <= 30]


    
    df['last_review'] = pd.to_datetime(df['last_review'])
    df.loc[df['last_review'].isna(), 'number_of_reviews'] = 0

    today = pd.to_datetime('2019-07-15')
    df['days_since_last_review'] = (today - df['last_review']).dt.days
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['days_since_last_review'] = df['days_since_last_review'].fillna(df['days_since_last_review'].max())

    
    df = df.dropna(subset=['price', 'minimum_nights', 'number_of_reviews', 'availability_365',
                           'longitude', 'latitude', 'room_type', 'neighbourhood_group'])
    return df

# Feature Engineering
def cap_outliers(df, columns=None, method="auto"):
    k = 1.5
    percentile_range = (1, 99)

    if isinstance(columns, str):
        columns = [columns]
    elif columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    capped_df = df.copy()

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Available columns: {df.columns.tolist()}")

        data = df[col].dropna()

        if method == "auto":
            if len(data) <= 5000:
                p_value = shapiro(data).pvalue
                is_normal = (p_value > 0.05) and (abs(skew(data)) < 0.5)
            else:
                is_normal = False
            method_ = "zscore" if is_normal else "iqr"
        else:
            method_ = method

        if method_ == "iqr":
            Q1, Q3 = np.percentile(data, [25, 75])
            IQR = Q3 - Q1
            cap_low = Q1 - k * IQR
            cap_high = Q3 + k * IQR
        elif method_ == "zscore":
            mean, std = data.mean(), data.std()
            cap_low = mean - k * std
            cap_high = mean + k * std
        elif method_ == "percentile":
            cap_low, cap_high = np.percentile(data, percentile_range)
        else:
            raise ValueError("Invalid method!")

        capped_df[col] = df[col].clip(cap_low, cap_high)

    return capped_df

def log_transform(df, columns):
    df[columns] = df[columns].apply(np.log1p)
    return df

def encoder_categorical(df, categorical_cols):
    existing_cat_cols = [col for col in categorical_cols if col in df.columns]
    if not existing_cat_cols:
        return df.values

    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(drop='first', sparse_output=False), existing_cat_cols)],
        remainder='passthrough'
    )
    return ct.fit_transform(df)

# KMeans Clustering
def simple_kmeans(X, k, max_iter=100, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids
    return labels, centroids

def bisecting_kmeans(X, k, max_iter=100, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Initialize with one cluster containing all points
    labels = np.zeros(len(X), dtype=int)
    centroids = [X.mean(axis=0)]
    
    while len(centroids) < k:
        # Calculate SSE for each existing cluster
        sse = []
        for i in range(len(centroids)):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                sse.append(np.sum((cluster_points - centroids[i])**2))
            else:
                sse.append(0)
        
        # Choose the cluster with largest SSE to split
        cluster_to_split = np.argmax(sse)
        cluster_points = X[labels == cluster_to_split]
        
        # Perform 2-means on the selected cluster
        if len(cluster_points) >= 2:
            split_labels, split_centroids = simple_kmeans(cluster_points, k=2, max_iter=max_iter)
            
            # Update labels and centroids
            new_centroid_idx = len(centroids)
            mask = (labels == cluster_to_split)
            
            # Update labels: 0 becomes cluster_to_split, 1 becomes new_centroid_idx
            labels[mask] = np.where(split_labels == 0, cluster_to_split, new_centroid_idx)
            
            # Update centroids
            centroids[cluster_to_split] = split_centroids[0]
            centroids.append(split_centroids[1])
        else:
            # Can't split clusters with <2 points
            break
    
    return labels, np.array(centroids)

def elbow_plot(X_scaled, method='kmeans'):
    inertias = []
    for k in range(1, 11):
        if method == 'kmeans':
            labels, centroids = simple_kmeans(X_scaled, k=k, random_state=42, max_iter=20)
        elif method == 'bisecting':
            labels, centroids = bisecting_kmeans(X_scaled, k=k, random_state=42, max_iter=20)
        inertia = np.sum((X_scaled - centroids[labels])**2)
        inertias.append(inertia)
    
    plt.figure(figsize=(8,5))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title(f'Elbow Method For Optimal k ({method})')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    plt.show()
    return inertias

def plot_clusters_pca(X_scaled, cluster_labels, method='kmeans'):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=components[:,0], y=components[:,1], hue=cluster_labels, palette='tab10', alpha=0.7)
    plt.title(f'Cluster Visualization after PCA ({method})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.show()

def clustering_pipeline(df, cluster_cols, title="", method='kmeans'):
    df = df.copy()
    features = df[cluster_cols]

    cat_cols_to_encode = [col for col in CATEGORICAL_COLS if col in cluster_cols]
    geo_cols_included = [col for col in GEO_COLS if col in cluster_cols]
    av_col_included = [col for col in AVAILABILITY if col in cluster_cols]
    non_geo_numeric_cols = [col for col in cluster_cols 
                          if col not in GEO_COLS 
                          and col not in CATEGORICAL_COLS
                           and col not in AVAILABILITY]

    # Encode categorical (returns either DataFrame or array)
    features_encoded = encoder_categorical(features, cat_cols_to_encode) if cat_cols_to_encode else None

    # Handle categorical (unscaled)
    X_categorical = features_encoded if features_encoded is not None else np.empty((len(df), 0))
    if isinstance(X_categorical, pd.DataFrame):
        X_categorical = X_categorical.values  # Convert to NumPy array if needed

    # Scale numeric non-geo columns
    X_non_geo_scaled = (
        RobustScaler().fit_transform(features[non_geo_numeric_cols]) 
        if non_geo_numeric_cols 
        else np.empty((len(df), 0)))
    
    # Scale geo columns
    X_geo_scaled = (
        StandardScaler().fit_transform(df[geo_cols_included]) 
        if geo_cols_included 
        else np.empty((len(df), 0))
    )
    X_av_scaled = (MinMaxScaler().fit_transform(df[av_col_included])
                if av_col_included 
                else np.empty((len(df), 0)))

    # Combine all features
    X_scaled = np.hstack([X_categorical, X_non_geo_scaled, X_geo_scaled, X_av_scaled])

    inertias = elbow_plot(X_scaled, method=method)

    kn = KneeLocator(range(1, 11), inertias, curve='convex', direction='decreasing')
    optimal_k = kn.knee
    print(f"Optimal k: {optimal_k}")

    if method == 'kmeans':
        final_labels, final_centroids = simple_kmeans(X_scaled, optimal_k, random_state=42, max_iter=10)
    elif method == 'bisecting':
        final_labels, final_centroids = bisecting_kmeans(X_scaled, optimal_k, random_state=42, max_iter=10)
    else:
        raise ValueError("Invalid clustering method. Choose 'kmeans' or 'bisecting'")
    
    df['Cluster'] = final_labels

    df[LOG_COLS] = df[LOG_COLS].apply(np.expm1)

    analysis = cluster_analysis(df, df['Cluster'])

    print(f"\n=== Cluster Profiles ({method}) ===")
    print(analysis['cluster_profiles'])

    print(f"\n=== Review Activity ({method}) ===")
    print(analysis['review_activity'])

    plot_clusters_pca(X_scaled, df['Cluster'], method=method)

    return df, analysis

# Cluster Analysis
def cluster_analysis(df, labels):
    df = df.copy()
    df['Cluster'] = labels

    def get_mode(series):
        modes = series.mode()
        return modes[0] if not modes.empty else np.nan

    numeric_stats = df.groupby('Cluster').agg({
        
        'price': ['min', 'max', 'median', 'mean', 'std', get_mode],
        'minimum_nights': ['min', 'max', 'median', 'mean', get_mode],
        'number_of_reviews': ['min', 'max', 'mean', 'sum'],
        'reviews_per_month': ['min', 'max', 'mean', 'max'],
        'availability_365': ['min', 'max', 'mean', 'median'],
        'days_since_last_review': ['min', 'max', 'mean', 'median']
    }).round(2)

    categorical_stats = df.groupby('Cluster').agg({
        'room_type': lambda x: x.mode()[0],
        'neighbourhood_group': lambda x: x.mode()[0],
    })

    review_status = np.where(df['days_since_last_review'] == -1, 'No Reviews',
                          np.where(df['days_since_last_review'] <= 90, 'Recent (≤90d)',
                          np.where(df['days_since_last_review'] <= 365, 'Older (91-365d)', 
                                  'Inactive (>1yr)')))
    df['review_status'] = review_status
    review_dist = pd.crosstab(df['Cluster'], df['review_status'], normalize='index').round(2)

    profiles = []
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        profile = {
            'Cluster': cluster,
            'Size': f"{len(cluster_data):,} ({len(cluster_data)/len(df):.1%})",
            'Avg_Price': f"${cluster_data['price'].median():.0f}",
            'Min_Price': f"${cluster_data['price'].min():.0f}",
            'Max_Price': f"${cluster_data['price'].max():.0f}",
            'Usual_price': f"${get_mode(cluster_data['price']):.0f}",
            'Most_minimum_nights': f"{get_mode(cluster_data['minimum_nights']):.0f}",
            'number_of_reviews': f"{get_mode(cluster_data['number_of_reviews']):.0f}",
            'Dominant_Room_Type': cluster_data['room_type'].mode()[0],
            'Dominant_Neighbourhood': cluster_data['neighbourhood_group'].mode()[0],
            'Avg_Availability': f"{cluster_data['availability_365'].mean():.0f} days",
            'Most_Review_Activity_Status': cluster_data['review_status'].mode()[0]
        }
        profiles.append(profile)

    profile_df = pd.DataFrame(profiles).set_index('Cluster')

    return {
        'numeric_stats': numeric_stats,
        'categorical_stats': categorical_stats,
        'review_activity': review_dist,
        'cluster_profiles': profile_df
    }


# Main Program
def main():
    airbnb_df = load_and_clean_data('airbnb.csv')
    airbnb_df = cap_outliers(airbnb_df, columns=['price', 'number_of_reviews','days_since_last_review','reviews_per_month','days_since_last_review'], method="auto")
    airbnb_df = log_transform(airbnb_df, LOG_COLS)

    airbnb_df_BI = airbnb_df.copy()

    selected_cols = [
        'price', 'minimum_nights','number_of_reviews','days_since_last_review',
        'availability_365','longitude','latitude','room_type','neighbourhood_group','days_since_last_review'
    ]

    print("Standard K-Means Clustering:")
    clean_result, clean_analysis = clustering_pipeline(airbnb_df, selected_cols, "Cleaned Data", method='kmeans')

    print("\nBisecting K-Means Clustering:")
    bisect_result, bisect_analysis = clustering_pipeline(airbnb_df_BI, selected_cols, "Cleaned Data", method='bisecting')

    # Plot geographic distribution for both methods
    plt.figure(figsize=(16,6))
    
    plt.subplot(1,2,1)
    sns.scatterplot(
        x=airbnb_df['longitude'],
        y=airbnb_df['latitude'],
        hue=clean_result['Cluster'], palette='viridis',
        alpha=0.5
    )
    plt.title('Geographic Cluster Distribution (Standard K-Means)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Cluster')
    
    plt.subplot(1,2,2)
    sns.scatterplot(
        x=airbnb_df['longitude'],
        y=airbnb_df['latitude'],
        hue=bisect_result['Cluster'], palette='viridis',
        alpha=0.5
    )
    plt.title('Geographic Cluster Distribution (Bisecting K-Means)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Cluster')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()