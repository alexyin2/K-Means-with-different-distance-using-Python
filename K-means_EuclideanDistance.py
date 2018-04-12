import pandas as pd
import numpy as np
from pprint import pprint
import copy
import matplotlib.pyplot as plt
"""
# Different Data Sets that may also be educational for learning clustering: 
# https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling
"""
# 1. Data Import
Customer = pd.read_csv('/Users/alex/Desktop/CustomerClustering.csv')
Customer.head(n=5)
print(Customer.shape)  # 5 variables and 60 rows
v1 = Customer['Visit.Time'].values  # variable 1
v2 = Customer['Average.Expense'].values  # variable 2
v3 = Customer['Age'].values  # variable 3
X = np.array(list(zip(v1, v2, v3)))
pprint(X)

# 2. Data Visualization
# Sex Distribution
sex = [1, 0]
plt.bar((1, 0), pd.value_counts(Customer['Sex']))
plt.xticks(sex, ['Male', 'Female'])
plt.axis([-0.5, 1.5, 0, 45])
plt.show()

# Age Distribution
Customer['Age'].sort_values()
TF1 = Customer['Age'] < 16
Interval1 = Customer['Age'][TF1].count()
TF2 = (Customer['Age'] >= 16) & (Customer['Age'] < 20)
Interval2 = Customer['Age'][TF2].count()
TF3 = (Customer['Age'] >= 20) & (Customer['Age'] < 30)
Interval3 = Customer['Age'][TF3].count()
TF4 = (Customer['Age'] >= 30) & (Customer['Age'] < 40)
Interval4 = Customer['Age'][TF4].count()
TF5 = Customer['Age'] >= 40
Interval5 = Customer['Age'][TF5].count()

labels = ['PRE-TEEN', 'TEENAGER', "20's", "30's", "40's"]
sizes = [Interval1, Interval2, Interval3, Interval4, Interval5]
colors = ['blue', 'green', 'red', 'teal', 'brown']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.show()

# Age vs. Average.Expense Scatter Plot
Age = Customer['Age'].values
AverageExpense = Customer['Average.Expense'].values
plt.scatter(Age, AverageExpense)
plt.xlabel('Age')
plt.ylabel('Average Expense')
plt.show()


# 3. Define Distance: Euclidean Distance
def dist(l, m, ax):
    return np.linalg.norm(l - m, axis=ax)  # The meaning of axis=1 can be find on P.S


# 4. Clustering Preparation
# Number of clusters
# Here is just an example of how we choose the centroids first.
np.random.seed(26)
k = 3
# X coordinates of random centroids
C_x = np.random.randint(np.min(v1), np.max(v1), size=k)
# Y coordinates of random centroids
C_y = np.random.randint(np.min(v2), np.max(v2), size=k)
# Z coordinates of random centroids
C_z = np.random.randint(np.min(v3), np.max(v3), size=k)
C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)
pprint(C)

# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
# Clustering Lables
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)

# 5. Design the number of clusters(k): Elbow Method
"""
Here I'll try to use elbow method to decide the number of clusters.
But due to the fact that elbow method is not working, perhaps deciding clusters numbers 
based on strategy may also be a good solution.
"""
cost_function = np.zeros([100, 8])
# Loop will run till the error becomes zero
for q in range(2, 10):
    for a in range(100):
        np.random.seed(a)
        k = q
        C_x = np.random.choice(v1, size=k)
        # Y coordinates of random centroids
        C_y = np.random.choice(v2, size=k)
        # Z coordinates of random centroids
        C_z = np.random.choice(v3, size=k)
        C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)
        error = 1  # if error = 0, then the while loop won't execute
        while error != 0:  # if error = 0, means that all the centroids are the same as the previous
            # Assigning each value to its closest cluster
            for i in range(len(X)):
                distances = dist(X[i], C, 1)
                cluster = np.argmin(distances)  # There is an explanation of how it works
                clusters[i] = cluster
            # Storing the old centroid values
            C_old = copy.deepcopy(C)  # There is an explanation of the meaning of deepcopy
            # Finding the new centroids by taking the average value
            for i in range(k):
                points = [X[j] for j in range(len(X)) if clusters[j] == i]
                C[i] = np.mean(points, axis=0)
            error = dist(C, C_old, None)
        for b in range(len(X)):
            if clusters[b] == 0:
                c = [dist(X[b], C[0], 0)]
            elif clusters[b] == 1:
                c = [dist(X[b], C[1], 0)]
            elif clusters[b] == 2:
                c = [dist(X[b], C[2], 0)]
            elif clusters[b] == 3:
                c = [dist(X[b], C[3], 0)]
            elif clusters[b] == 4:
                c = [dist(X[b], C[4], 0)]
            elif clusters[b] == 5:
                c = [dist(X[b], C[5], 0)]
            elif clusters[b] == 6:
                c = [dist(X[b], C[6], 0)]
            elif clusters[b] == 7:
                c = [dist(X[b], C[7], 0)]
            else:
                c = [dist(X[b], C[8], 0)]
            cost_function[a, q-2] += c

mean_cost_function = pd.DataFrame(cost_function).mean()
mean_cost_function = list(mean_cost_function)
number_of_clusters = range(2, 10)

plt.title('Mean Cost Function in different clusters')
plt.axis([1.5, 9.5, 0, 600])
plt.bar(number_of_clusters, mean_cost_function, color='blue')
plt.show()
"""
By the elbow method, we may finally choose the clusters to be four.

In Contrast, we may choose clusters to be three due to marketing strategy.
"""
# 6. Run KMeans and doing random initialization (k = 3)
# Create an array that stores the cost function under different centroids
cost_function = np.zeros([100, 1])
# Loop will run till the error becomes zero
for a in range(100):
    np.random.seed(a)
    k = 3
    C_x = np.random.choice(v1, size=k)
    # Y coordinates of random centroids
    C_y = np.random.choice(v2, size=k)
    # Z coordinates of random centroids
    C_z = np.random.choice(v3, size=k)
    C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)
    error = 1  # if error = 0, then the while loop won't execute
    while error != 0:  # if error = 0, means that all the centroids are the same as the previous
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C, 1)
            cluster = np.argmin(distances)  # There is an explanation of how it works
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = copy.deepcopy(C)  # There is an explanation of the meaning of deepcopy
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
    for b in range(len(X)):
        if clusters[b] == 0:
            c = [dist(X[b], C[0], 0)]
        elif clusters[b] == 1:
            c = [dist(X[b], C[1], 0)]
        else:
            c = [dist(X[b], C[2], 0)]
        cost_function[a] += c

# 7. Choose the centroids with the lowest cost function (k = 3)
# The lowest cost function: cost_function[np.argmin(cost_function)]
np.random.seed(np.argmin(cost_function))  # Use the centroids with the lowest cost function to run K-Means again.
k = 3
C_x = np.random.choice(v1, size=k)
C_y = np.random.choice(v2, size=k)
C_z = np.random.choice(v3, size=k)
C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)
error = 1  # if error = 0, means that all the centroids are the same as the previous
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C, 1)
        cluster = np.argmin(distances)  # There is an explanation of how it works
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = copy.deepcopy(C)  # There is an explanation of the meaning of deepcopy
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

# 8. Analyze the Outcome (k = 3)
Type0 = Customer[clusters == 0].drop('ID', 1)  # Here we delete column 'ID' since it can't offer any information
Type1 = Customer[clusters == 1].drop('ID', 1)
Type2 = Customer[clusters == 2].drop('ID', 1)
print('Type0 length: %d; Type1 length: %d; Type2 length: %d' % ((len(Type0)), (len(Type1)), (len(Type2))))
Type0.mean()  # Type0: 兒童，消費能力不高
pd.value_counts(Type0['Age'], ascending=False)
pd.pivot_table(Type0, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Type1.mean()  # Type1: 主要為青少年，年齡大多在16~22之間，有能力自行支配金錢
pd.value_counts(Type1['Age'], ascending=False)
pd.pivot_table(Type1, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Type2.mean()  # Type2: 對該商家黏著度高，造訪次數高，購買能力強，年齡平均約30歲，其中有兩個購買能力強的青少年
pd.value_counts(Type2['Age'], ascending=False)
pd.pivot_table(Type2, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Customer['Cluster'] = clusters
Customer.sort_values(by=['Age', 'Cluster'], ascending=True)
Customer.sort_values(by=['Cluster', 'Average.Expense', 'Age'], ascending=[True, True, False])

# 9. Run KMeans and doing random initialization (k = 4)
# Create an array that stores the cost function under different centroids
cost_function = np.zeros([100, 1])
# Loop will run till the error becomes zero
for a in range(100):
    np.random.seed(a)
    k = 4
    C_x = np.random.choice(v1, size=k)
    # Y coordinates of random centroids
    C_y = np.random.choice(v2, size=k)
    # Z coordinates of random centroids
    C_z = np.random.choice(v3, size=k)
    C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)
    error = 1  # if error = 0, then the while loop won't execute
    while error != 0:  # if error = 0, means that all the centroids are the same as the previous
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C, 1)
            cluster = np.argmin(distances)  # There is an explanation of how it works
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = copy.deepcopy(C)  # There is an explanation of the meaning of deepcopy
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
    for b in range(len(X)):
        if clusters[b] == 0:
            c = [dist(X[b], C[0], 0)]
        elif clusters[b] == 1:
            c = [dist(X[b], C[1], 0)]
        elif clusters[b] == 2:
            c = [dist(X[b], C[2], 0)]
        else:
            c = [dist(X[b], C[3], 0)]
        cost_function[a] += c

# 10. Choose the centroids with the lowest cost function (k = 4)
# The lowest cost function: cost_function[np.argmin(cost_function)]
np.random.seed(np.argmin(cost_function))  # Use the centroids with the lowest cost function to run K-Means again.
k = 4
C_x = np.random.choice(v1, size=k)
C_y = np.random.choice(v2, size=k)
C_z = np.random.choice(v3, size=k)
C = np.array(list(zip(C_x, C_y, C_z)), dtype=np.float32)
error = 1  # if error = 0, means that all the centroids are the same as the previous
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C, 1)
        cluster = np.argmin(distances)  # There is an explanation of how it works
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = copy.deepcopy(C)  # There is an explanation of the meaning of deepcopy
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

# 11. Analyze the Outcome (k = 4)
Type0 = Customer[clusters == 0].drop('ID', 1)  # Here we delete column 'ID' since it can't offer any information
Type1 = Customer[clusters == 1].drop('ID', 1)
Type2 = Customer[clusters == 2].drop('ID', 1)
Type3 = Customer[clusters == 3].drop('ID', 1)
print('Type0 length: %d; Type1 length: %d; Type2 length: %d; Type 3 length: %d' %
      ((len(Type0)), (len(Type1)), (len(Type2)), (len(Type3))))

Type0.mean()  # Type0:
pd.value_counts(Type0['Age'], ascending=False)
pd.pivot_table(Type0, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Type1.mean()  # Type1:
pd.value_counts(Type1['Age'], ascending=False)
pd.pivot_table(Type1, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Type2.mean()  # Type2:
pd.value_counts(Type2['Age'], ascending=False)
pd.pivot_table(Type2, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Type3.mean()  # Type3:
pd.value_counts(Type3['Age'], ascending=False)
pd.pivot_table(Type3, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Customer['Cluster'] = clusters
Customer.sort_values(by=['Age', 'Cluster'], ascending=True)
Customer.sort_values(by=['Cluster', 'Average.Expense', 'Age'], ascending=[True, True, False])

"""
More thoughts that can make this project better.
Is there any way to improve the calculation of KMeans. 
E.g., mean only need to calculate the data's that has changed.
"""

# P.S
# The meaning of np.linalg.norm()

a = (np.arange(9) - 4).reshape((3, 3))
b = (np.arange(9) - 12).reshape((3, 3))
print(a, b, sep='\n')
np.linalg.norm(a - b, axis=None)  # same as math.sqrt(64 * 9)
np.linalg.norm(a - b, axis=1)  # same as math.sqrt(64 * 3)

# The meaning of np.argmin()
a = np.array([[6, 5, 9],
              [8, 7, 6],
              [5, 6, 1]])
np.argmin(a, axis=0)  # Shows the smallest index in each column
np.argmin(a, axis=1)  # Shows the smallest index in each row

b = np.array([[2],
              [5],
              [9],
              [1]])
np.argmin(b)  # It tells which row is the smallest

# The meaning of deepcopy()
"""
The difference of copy and deepcopy is that a variable which is produced by deepcopy is independent.
But a variable which is produced by copy is still related to the original data
"""
origin = [1, 2, [3, 4]]
by_copy = copy.copy(origin)
by_deepcopy = copy.deepcopy(origin)
print(by_copy, by_deepcopy, sep='\n')
origin[2][0] = 'hey'
print(by_copy, by_deepcopy, sep='\n')
