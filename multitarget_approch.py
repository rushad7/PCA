import time
start_time = time.time()

import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import_time = time.time() - start_time
print(import_time)

pose_current = np.array( [ [7,8,12,13,14,17,18,19,23,24,28,29], [8,9,12,13,14,17,18,23,29,32,33,34], [7,8,12,13,14,17,18,19,23,24,28,29], [8,9,12,13,14,17,18,23,29,32,33,34] ] )
pose_future = np.array( [ [8,9,12,13,14,17,18,23,29,32,33,34], [7,8,12,13,14,17,18,19,23,24,28,29], [8,9,12,13,14,17,18,23,29,32,33,34], [7,8,12,13,14,17,18,19,23,24,28,29] ] )

reg_model = MultiOutputRegressor(AdaBoostRegressor(), n_jobs=-1)

pca = PCA(n_components=1)
x_pca = pca.fit_transform(pose_current)
y_pca = pca.fit_transform(pose_future)

x_current = np.array([[8,9,12,13,14,17,18,23,29,32,33,34]])
x_current_pca = pca.fit_transform(x_current)

reg_model.fit(x_pca, y_pca)
predicted_pose_pca = reg_model.predict(x_current_pca)
predicted_pose = pca.inverse_transform(predicted_pose_pca)

run_time = time.time() - import_time - start_time
print(run_time)

#axes = plt.axes()
#axes.set_ylim([-0.001, 0.001])
#plt.plot(predicted_pose[0] - pose_future[0], 'r')
#plt.show()
x_pos = []
y_pos = []
for i in range(len(pose_current[0])):
    x_pos.append(round(pose_current[0][i]/5))
    y_pos.append(round((pose_current[0][i]%5)))
plt.plot(x_pos, y_pos, 'ro')
plt.plot(x_pos, y_pos, 'b--')
plt.show()
