import numpy as np
from sklearn.cluster import KMeans
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor

df = pandas.read_csv("Final.csv")
columns = list(df.columns.values)
scaler = StandardScaler()
npdf = scaler.fit(df.to_numpy()).transform(df.to_numpy())
df = pandas.DataFrame(npdf, columns=columns)

lf = df.drop(columns=["X", "Y", "VWC"])
values = list(lf.columns.values)

actual = []
predicted = []
all_models = []
all_classes = []

np.random.seed(22)
np.random.set_state(np.random.get_state())

# Sample data (replace with your data)
response_variable = np.squeeze(df[["VWC"]].to_numpy())
external_variable = df[["DEM_QGIS", "PlnCurv", "TWI"]].to_numpy()
coordinates = df[["X", "Y"]].to_numpy()

# Perform KMeans clustering on the external variable
np.random.seed(23)
np.random.set_state(np.random.get_state())
kmeans = KMeans(n_clusters=7)
labels = kmeans.fit_predict(np.hstack((external_variable, coordinates)))
num_classes = len(np.unique(labels))

for i in range(0, len(response_variable)):
    np.random.seed(22)
    np.random.set_state(np.random.get_state())
    response_variable_test = np.array([response_variable[i]])
    response_variable_train = np.delete(response_variable, i, axis=0)
    external_variable_test = np.array([external_variable[i]])
    external_variable_train = np.delete(external_variable, i, axis=0)
    coordinates_test = np.array([coordinates[i]])
    coordinates_train = np.delete(coordinates, i, axis=0)
    labels_test = np.array([labels[i]])
    labels_train = np.delete(labels, i, axis=0)
    print(labels_test)


    NN_models = []
    NN_models_label = []

    for class_label in np.unique(labels_train):
        class_idx = np.where(labels_train == class_label)[0]
        class_response = response_variable_train[class_idx]
        class_external = external_variable_train[class_idx]
        class_corrdinates = coordinates_train[class_idx]

        np.random.seed(22)
        np.random.set_state(np.random.get_state())
        NN = MLPRegressor(hidden_layer_sizes=(5,100))
        NN.fit(np.hstack((class_external, class_corrdinates)), class_response)

        NN_models.append(NN)
        NN_models_label.append(class_label)

    predictions = []

    
    for i in range(0,len(labels_test)):
        catagory = labels_test[i]
        NN = NN_models[catagory]

        inp = external_variable_test[i,0:len(["DEM_QGIS", "PlnCurv", "TWI"])]

        try:
            prediction = NN.predict(np.hstack((inp, np.array([coordinates_test[i,:]]))))
        except ValueError:
            inp = np.array([inp])
            prediction = NN.predict(np.hstack((inp, np.array([coordinates_test[i,:]]))))

        predictions.append(prediction[0])

    actual.append(response_variable_test)
    predicted.append(predictions)
    print(response_variable_test[0] - predictions[0])
    all_models.append(NN_models)
    all_classes.append(NN_models_label)


print(r2_score(np.concatenate(actual),np.concatenate(predicted)))