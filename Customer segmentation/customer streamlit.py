import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('ifood_df.csv')
    df.drop(columns=['Z_CostContact','Z_Revenue'], inplace=True)
    return df

df = load_data()

# Data Preparation
X = df[['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]
X_normalized = (X - X.mean()) / X.std()

# Clustering
def perform_clustering(X, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans

# Sidebar for number of clusters
k = st.sidebar.slider('Number of clusters', 1, 10, 3)

clusters, kmeans = perform_clustering(X_normalized, k)
df['Cluster'] = clusters

# Elbow Method Plot
st.subheader('Elbow Method')
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_normalized)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss)
ax.set_title('Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Scatter Plot
st.subheader('Customer Segmentation Scatter Plot')
fig, ax = plt.subplots()
scatter = ax.scatter(df['MntWines'], df['MntMeatProducts'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel('Wine Purchases')
ax.set_ylabel('Meat Purchases')
ax.set_title('Customer Segmentation')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
st.pyplot(fig)

# Bar Plots for each attribute
st.subheader('Cluster Attributes')
def plot_cluster_bar(df, feature):
    plt.figure(figsize=(4, 2))
    sns.barplot(data=df, x='Cluster', y=feature)
    plt.title(f'{feature} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Mean')
    plt.xticks(range(k), [f'Cluster {cluster}' for cluster in range(k)])
    return plt

for column in X.columns:
    st.pyplot(plot_cluster_bar(df, column))

# User Input Form
st.sidebar.header('Input Customer Data')
def user_input_features():
    income = st.sidebar.number_input('Income', min_value=0)
    kidhome = st.sidebar.number_input('Kidhome', min_value=0, max_value=10)
    teenhome = st.sidebar.number_input('Teenhome', min_value=0, max_value=10)
    recency = st.sidebar.number_input('Recency', min_value=0)
    mnt_wines = st.sidebar.number_input('MntWines', min_value=0)
    mnt_fruits = st.sidebar.number_input('MntFruits', min_value=0)
    mnt_meat_products = st.sidebar.number_input('MntMeatProducts', min_value=0)
    mnt_fish_products = st.sidebar.number_input('MntFishProducts', min_value=0)
    mnt_sweet_products = st.sidebar.number_input('MntSweetProducts', min_value=0)
    mnt_gold_prods = st.sidebar.number_input('MntGoldProds', min_value=0)
    num_deals_purchases = st.sidebar.number_input('NumDealsPurchases', min_value=0)
    num_web_purchases = st.sidebar.number_input('NumWebPurchases', min_value=0)
    num_catalog_purchases = st.sidebar.number_input('NumCatalogPurchases', min_value=0)
    num_store_purchases = st.sidebar.number_input('NumStorePurchases', min_value=0)
    num_web_visits_month = st.sidebar.number_input('NumWebVisitsMonth', min_value=0)

    data = {
        'Income': income,
        'Kidhome': kidhome,
        'Teenhome': teenhome,
        'Recency': recency,
        'MntWines': mnt_wines,
        'MntFruits': mnt_fruits,
        'MntMeatProducts': mnt_meat_products,
        'MntFishProducts': mnt_fish_products,
        'MntSweetProducts': mnt_sweet_products,
        'MntGoldProds': mnt_gold_prods,
        'NumDealsPurchases': num_deals_purchases,
        'NumWebPurchases': num_web_purchases,
        'NumCatalogPurchases': num_catalog_purchases,
        'NumStorePurchases': num_store_purchases,
        'NumWebVisitsMonth': num_web_visits_month
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combine input data with the dataset for clustering
combined_df = pd.concat([df, input_df], axis=0)

# Normalize the combined data
X_combined = combined_df[['Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                          'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                          'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                          'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']]
X_combined_normalized = (X_combined - X_combined.mean()) / X_combined.std()

# Apply clustering
combined_clusters, _ = perform_clustering(X_combined_normalized, k)
combined_df['Cluster'] = combined_clusters

# Show the input data
st.subheader('User Input Data')
st.write(input_df)

# Show the cluster of the input data
st.subheader('Predicted Cluster for Input Data')
st.write(combined_df.tail(1)[['Cluster']])
