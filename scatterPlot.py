import matplotlib.pyplot as plt
import pandas
import numpy as np

#classification = ["AC", "D", "GM", "KMeans", "MS"]
classification = ["KMeans"]
classification_labels = ["KMeans"]
#regression = ["GB", "NN", "RF", "SVM", "UniversalKriggingWithDrift"]
regression = ["NN"]
regression_labels = ["Neural Network"]
colors = ["red", "blue", "green", "orange", "purple"]

for c in classification:
    for r, color in zip(regression, colors):
        data_file = "RawData/" + c + "_" + r + ".csv"
        data_frame = pandas.read_csv(data_file)

        predicted_data = data_frame['Predicted'].to_numpy().astype(np.float64)
        actual_data = data_frame['Actual'].to_numpy().astype(np.float64)

        regression_label = regression_labels[regression.index(r)]

        plt.scatter(predicted_data, actual_data, c=color, label=regression_label)
        slope, intercept = np.polyfit(predicted_data, actual_data, 1)  # Calculate slope and intercept of the linear fit
        plt.plot(np.array(np.arange(.25, .5+.0125, .0125)), slope * np.arange(.25, .5+.0125, .0125) + intercept, linewidth=2)
    
    classification_label = classification_labels[classification.index(c)]

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Actual VS Predicted Volumetric Water Content with\n" + classification_label + " Classification")
    plt.xlim(.25, .5)
    plt.ylim(.25, .5)
    plt.legend()

    output_file = 'Plots/' + classification_label
    plt.savefig(output_file+".tiff", format='tiff', dpi=300)
    plt.savefig(output_file+".png", format='png', dpi=300)
  

    plt.clf()