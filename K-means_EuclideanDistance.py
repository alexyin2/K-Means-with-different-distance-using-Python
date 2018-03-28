import pandas as pd
import numpy as np
from pprint import pprint
import copy
import matplotlib.pyplot as plt

# 1. Data Import
Customer = pd.read_csv('CustomerClustering.csv')
Customer.head(n=5)
Customer.shape  # 5 variables and 60 rows
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
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'red']
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
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)  # The meaning of axis=1 can be find on P.S


# 4. Clustering Preparation
# Number of clusters
np.random.seed(62)
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

# 5. Run KMeans
# Loop will run till the error becomes zero
while error != 0:  # if error = 0, means that all the centroids are the same as the previous
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)  # There is an explanation of how it works
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = copy.deepcopy(C)  # There is an explanation of the meaning of deepcopy
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)


# 6. Analyze the Outcome
print(C)
Type0 = Customer[clusters == 0].drop('ID', 1)  # Here we delete column 'ID' since it can't offer any information
Type1 = Customer[clusters == 1].drop('ID', 1)
Type2 = Customer[clusters == 2].drop('ID', 1)
print('Type0 length: %d; Type1 length: %d; Type2 length: %d' % ((len(Type0)), (len(Type1)), (len(Type2))))
Type0.mean()  # Type0: 對該商家黏著度高，可以看造訪次數或花費是不是都多於某數值，年齡分佈大多高於20，少數幾個狂熱年輕人
pd.value_counts(Type0['Age'], ascending=False)
pd.pivot_table(Type0, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Type1.mean()  # Type1: 一般顧客，年齡大多在20~30之間，較有能力自行支配金額
pd.value_counts(Type1['Age'], ascending=False)
pd.pivot_table(Type1, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Type2.mean()  # Type2: 兒童
pd.value_counts(Type2['Age'], ascending=False)
pd.pivot_table(Type2, index=['Sex'], values=['Visit.Time', 'Average.Expense', 'Age', 'Sex'],
               aggfunc={'Visit.Time': np.mean, 'Average.Expense': np.mean,
                        'Age': np.mean, 'Sex': pd.value_counts})

Customer['Cluster'] = clusters
Customer.sort_values(by=['Age', 'Cluster'], ascending=True)
Customer.sort_values(by=['Cluster', 'Average.Expense'], ascending=[True, True])


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
