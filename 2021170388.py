import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#upload data excel sheet
data = pd.read_csv('assignment1dataset.csv')
print(data.describe())

#split data
x1=data['Hours Studied']
x2=data['Previous Scores']
x3=data['Sleep Hours']
x4=data['Sample Question Papers Practiced']
y=data['Performance Index']

#*************** first feature Hours studied ******************

#visualize data between feature hours studied and performance

plt.scatter(x1, y)
plt.xlabel('Hours Studied', fontsize = 20)
plt.ylabel('Performance Index', fontsize = 20)
plt.show()

#constants for the first feature
L1 = 0.01
epochs1 = 3000
m1=0
c1=0
length_of_x1= float(len(x1))

#the iteration to find best m and c make the least error

for i in range(epochs1):
    y_prediction1 = m1*x1 + c1
    dm1= (-2/length_of_x1) * sum((y -  y_prediction1)* x1)
    dc1 = (-2/length_of_x1) * sum(y -  y_prediction1)
    m1 = m1 - L1* dm1
    c1 = c1 - L1 * dc1

prediction1 = m1*x1 + c1

#the best line and mean square error for first feature =317

plt.scatter(x1, y)
plt.xlabel('Hours Studied', fontsize = 20)
plt.ylabel('Performance Index', fontsize = 20)
plt.plot(x1, prediction1, color='red', linewidth = 3)
plt.show()

print('Mean Square Error', metrics.mean_squared_error(y, prediction1))


#********************* second feature Previous Scores *****************************

#visualize second feature Previous Scores
plt.scatter(x2, y)
plt.xlabel('Previous Scores', fontsize = 20)
plt.ylabel('Performance Index', fontsize = 20)
plt.show()

#constants of second feature
L2 = 0.0001
epochs2 = 400000
m2=0
c2=0
length_of_x2= float(len(x2))

#iteration to find best m and c for second feature

for i in range(epochs2):
    y_prediction2 = m2*x2 + c2
    dm2= (-2/length_of_x2) * sum((y -  y_prediction2)* x2)
    dc2 = (-2/length_of_x2) * sum(y -  y_prediction2)
    m2 = m2 - L2* dm2
    c2 = c2 - L2 * dc2

prediction2 = m2*x2 + c2

#the best line and mean square error for second feature Previous Scores

plt.scatter(x2, y)
plt.xlabel('Previous Scores', fontsize = 20)
plt.ylabel('Performance Index', fontsize = 20)
plt.plot(x2, prediction2, color='red', linewidth = 3)
plt.show()

print('Mean Square Error', metrics.mean_squared_error(y, prediction2))


#********************** third feature Sleep Hours **************************

#visualize third feature sleep Hours

plt.scatter(x3, y)
plt.xlabel('Sleep Hours', fontsize = 20)
plt.ylabel('Performance Index', fontsize = 20)
plt.show()

#constants of third feature

L3 = 0.01
epochs3 = 5000
m3=0
c3=0
length_of_x3= float(len(x3))

#the iteration to find the best m and c for the third feature

for i in range(epochs3):
    y_prediction3 = m3 * x3 + c3
    dm3 = (-2 / length_of_x3) * sum((y - y_prediction3) * x3)
    dc3 = (-2 / length_of_x3) * sum(y - y_prediction3)
    m3 = m3 - L3 * dm3
    c3 = c3 - L3 * dc3

prediction3 = m3*x3 + c3

 #the best line and the mean square error for sleep Hours feature

plt.scatter(x3, y)
plt.xlabel('Sleep Hours', fontsize = 20)
plt.ylabel('Performance Index', fontsize = 20)
plt.plot(x3, prediction3, color='red', linewidth = 3)
plt.show()

print('Mean Square Error', metrics.mean_squared_error(y, prediction3))


#************************** fourth feature Sample Question Papers Practiced**************

#visualize feature four Sample Question Papers Practiced
plt.scatter(x4, y)
plt.xlabel('Sample Question Papers Practiced', fontsize = 20)
plt.ylabel('Performance Index', fontsize = 20)
plt.show()


#constants of feature four
L4 = 0.01
epochs4 = 2000
m4=0
c4=0
length_of_x4= float(len(x4))

#the iteration to find best m and c
for i in range(epochs4):
    y_prediction4 = m4*x4 + c4
    dm4= (-2/length_of_x4) * sum((y -  y_prediction4)* x4)
    dc4 = (-2/length_of_x4) * sum(y -  y_prediction4)
    m4 = m4 - L4* dm4
    c4 = c4 - L4 * dc4

prediction4 = m4*x4 + c4

#the best line fit and mean square error

plt.scatter(x4, y)
plt.xlabel('Sample Question Papers Practiced', fontsize = 20)
plt.ylabel('Performance Index', fontsize = 20)
plt.plot(x4, prediction4, color='red', linewidth = 3)
plt.show()

print('Mean Square Error', metrics.mean_squared_error(y, prediction4))