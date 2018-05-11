# Learning K-means in Python without using sklearn

* **Warning**: 
    - The code still has some problem when running the loop.
    - It sometimes will have a RunTimeError, but it doesn't effect the outcome I guess.

* I'm using Python to define different distance and run K-Means without using packages since it helps a lot in understanding the mathematics behind.

* Here, I've defined several different distances:
  1. Euclidean
  2. Manhattan
  3. ChebyShev distance
  4. A new distance defined by myself using weights which is decided based on domain knowledge.
  
* I've also done some data visualization at the beggining and look at the clusters I've generated.

* **Improvements for this repository**:
  - Data visualization should be clearer for showing the cluster outcome
  - Better introduction of the distance defined by myself
  - The way of choosing the first centroids can be improved
  
## New ideas about Kmeans
* I've participated in a class, named Learning from Data, in NCCU, and we've disgussed about the issues or problems we may met in running K means.
* I really recommend participating in one of the classes teached by Chang, Yuan-chin Ivan. He's an inspiring and enlightening professor.
* I'll post some of my notes here.  

1. Silhouette method: I should need to know the mathematics about how this works on deciding k.
2. Canberra distance: Instead of Euclidean Distance, this method is often used in our project.  
I should need to know the mathematics and which domain it fits on.
3. Naming the clusters: It's a theory of naming the clusters after we've run kmeans. [Reference](http://www.d.umn.edu/~tpederse/Pubs/cicling2005.pdf)
4. When introductin your project to others, we should avoid the action of going to the last page of our powerpoint of pdf.  
This means that we will have to work hard on how we describe our data and how we perform informations of our outcome, such as visualizing or naming.
5. What effect will highly correlated variables impact on running kmeans?  
How should we do to solve this problem? There's no real answer for this question and I'll say that it all based on different situations.
6. Is ordinal data suitable for running kmeans. The answer is not certain, but the professor gives his reason of why not putting ordinal data in kmeans model.  
It's because there's still no distance that deals well both on numerical and ordinal datas.  
The professor also suggested some actions that may be used when facing such situation.  
First of all, we can just change all the numerical data into ordinal, but is this really meaningful and effective? 
Second, we may use our domain knowledge to change ordinal data into numerical data. For example, perhaps our data has 4 levels in performing on tests: worse, bad, normal, good. Maybe we can turn them into numbers like: 30, 60, 75, 90. 
