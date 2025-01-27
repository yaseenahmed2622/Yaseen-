# Yaseen-1. TransactionID: Unique ID for each transaction.


2. Date: Date of the transaction.


3. CustomerID: Unique ID for each customer.


4. ProductCategory: The category of the product sold (e.g., Electronics, Clothing).


5. SalesAmount: Total sales amount for the transaction.


6. Quantity: Number of items purchased.


7. Region: The region where the sale occurred (e.g., North, South)






2) 2)1. CustomerID: C0001

Lookalike 1: C0033 (Score: 0.9562)

Lookalike 2: C0092 (Score: 0.8709)

Lookalike 3: C0030 (Score: 0.7373)



2. CustomerID: C0002

Lookalike 1: C0015 (Score: 0.9746)

Lookalike 2: C0066 (Score: 0.8073)

Lookalike 3: C0056 (Score: 0.7543)





3) 1: Import Necessary Libraries

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


---

Step 2: Load and Prepare Your Data

You’ll need two datasets: Customers.csv and Transactions.csv.

Customers.csv contains profile information like Age, Gender, AnnualIncome, and SpendingScore.

Transactions.csv contains transaction details like CustomerID, AmountSpent, and ProductCategory.


# Load datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Aggregate transaction data
transaction_agg = transactions.groupby('CustomerID').agg({
    'AmountSpent': 'mean',
    'TransactionID': 'count'
}).rename(columns={'AmountSpent': 'AvgTransactionAmount', 'TransactionID': 'TotalTransactions'})

# Merge customer and transaction data
merged_data = pd.merge(customers, transaction_agg, on='CustomerID', how='left').fillna(0)

# Encode categorical data (e.g., Gender and ProductPreference)
merged_data['Gender'] = merged_data['Gender'].map({'Male': 0, 'Female': 1})
merged_data = pd.get_dummies(merged_data, columns=['ProductPreference'])


---

Step 3: Scale Features

Normalize numerical data for clustering.

# Features for clustering
clustering_features = ['Age', 'Gender', 'AnnualIncome', 'SpendingScore', 'AvgTransactionAmount', 'TotalTransactions'] + \
                      [col for col in merged_data.columns if 'ProductPreference_' in col]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data[clustering_features])


---

Step 4: Apply Clustering

Use K-Means and choose an optimal number of clusters between 2 and 10.

# Apply K-Means with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the dataset
merged_data['Cluster'] = clusters

# Evaluate clustering metrics
db_index = davies_bouldin_score(scaled_data, clusters)
silhouette_avg = silhouette_score(scaled_data, clusters)

print(f"Davies-Bouldin Index: {db_index}")
print(f"Silhouette Score: {silhouette_avg}")


---

Step 5: Visualize Clusters

Reduce dimensions using PCA for visualization.

# Reduce dimensions with PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)
merged_data['PCA1'] = reduced_data[:, 0]
merged_data['PCA2'] = reduced_data[:, 1]

# Plot clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=merged_data, palette='viridis', s=100)
plt.title("Customer Segments")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title="Cluster")
plt.show()


---

Step 6: Save Results

Save the segmented dataset for further analysis.

merged_data.to_csv("Clustered_Customers.csv", index=False
