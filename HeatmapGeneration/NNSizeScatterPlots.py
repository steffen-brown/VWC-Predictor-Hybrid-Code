import matplotlib.pyplot as plt
import pandas
import numpy as np
from ast import literal_eval

data = pandas.read_csv("shape_comparision_results.csv")
np_data = data.to_numpy()
points_string = np_data[:,0].tolist()
points = [literal_eval(s) for s in points_string]

first_layer = [pos[0] for pos in points[19:]]
second_layer = [pos[1] for pos in points[19:]]
r2 = np_data[19:,1].tolist()

top_df = pandas.DataFrame({'Position': points[19:], 'Z Value': r2})
top_df = top_df.sort_values(by='Z Value', ascending=False).head(5)
print(top_df)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(first_layer, second_layer, r2, c='b', marker='o')

ax.set_zlim(0, .75)

ax.set_xlabel('First Layer Size')
ax.set_ylabel('Second Layer Size')
ax.set_zlabel('R2')

plt.show()