import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
url = 'data.csv'
data = pd.read_csv(url,skiprows=12)

# Filter the first 1500 rows and those with distance <= 4 Mpc
filtered_data = data['D (Mpc)'].head(1500)
filtered_data = filtered_data[filtered_data <= 4]

# Plot histogram with 10 bins
plt.figure(figsize=(8, 6))
n, bins, patches = plt.hist(filtered_data, bins=10, edgecolor='black')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Number of Objects')
plt.title('Histogram of Metric Distance (10 bins)')
plt.savefig('10binhistogram.png')

# Calculated and print estimated probabilities
total_points = len(filtered_data)
estimated_probabilities = n / total_points
print('Estimated probabilities for each bin:', estimated_probabilities)

def cross_validation_score(data, max_bins):
    scores = []
    n = len(data)  # Total number of points
    bin_width=[]
    for bins in range(1, max_bins + 1):
        # Histogram for the given bin count
        h=4/bins  # h is equal to 4/bins
        bin_width.append(h)
        hist, bin_edges = np.histogram(data, bins=bins)
        probability=hist/n   # calculated the probability
        sum_squared_probabilities=np.sum(probability**2)
        J_h = (2 / ((n - 1) * h)) - ((n + 1) / ((n - 1) * h)) * sum_squared_probabilities
        scores.append(J_h)
    return scores,bin_width



bin_range = range(1, 1001)
scores,bin_width = cross_validation_score(filtered_data, 1000)
# Plot the cross-validation scores
plt.figure(figsize=(8, 6))
plt.plot(bin_width, scores, color='blue')
plt.xlabel('Bin width(h)')
plt.ylabel('Cross-validation Score')
plt.title('Cross-validation Score vs bin width')
plt.savefig('crossvalidation.png')
plt.show()

optimal_bins = np.argmin(scores) + 1
print('Optimal number of bins:', optimal_bins)

plt.figure(figsize=(8, 6))
plt.hist(filtered_data, bins=optimal_bins, edgecolor='black')
plt.xlabel('Distance (Mpc)')
plt.ylabel('Number of Objects')
plt.title(f'Optimal Histogram with {optimal_bins} bins')
plt.savefig('optimalhistogram.png')
plt.show()