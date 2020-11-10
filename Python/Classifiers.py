import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn import tree, svm, neighbors, linear_model

# Variables
fig_height = 2
fig_width = 8
path = "dataset/"
images = os.listdir(path)
img_list = []
file_name_list = []
number_label_list = []

for img in images:
    # Load images into an Array
    img_list.append(cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE))

    # Parse file names
    file_name, extension = os.path.splitext(img)
    file_name_list.append(file_name)
    font_number, number_label = file_name.split(".")
    number_label_list.append(number_label)

max_length = len(img_list)
training_data_length = int(max_length * 0.8)
testing_data_length = int(max_length * 0.2)


# Split Array into Training and Testing dataset 
img_list = np.array(img_list)
reshaped_array = np.reshape(img_list, (max_length, 256))

training_data = reshaped_array[0:training_data_length]
training_label = number_label_list[0:training_data_length]

testing_data = reshaped_array[training_data_length:]
actual_label = number_label_list[training_data_length:]

predicted_values = []

def plot_images(predicted_values):
    # Create plot
    offset = 0
    fig = plt.figure()
    for i in range(fig_height):
        for j in range(fig_width):
            plt.subplot2grid((fig_height, fig_width), (i, j))
            plt.imshow(img_list[testing_data_length+j+offset], cmap="gray")
            plt.axis('off')
            plt.title('Predicted number = ' + str(predicted_values[j+offset]),fontdict={'fontsize': 8})
        offset += fig_width
    plt.show()

def predict_image(classifier, predicted_values):
    # Predict Images
    for img in range(testing_data_length):
        predicted_image=testing_data[img]
        reshaped = predicted_image.reshape(16,16)
        predicted_values.append(classifier.predict([testing_data[img]]))
        
    # Change output array type to string
    predicted_values = np.array([x[0].astype(str) for x in predicted_values])
    return predicted_values

def calculate_accuracy(predicted_values):
    count = 0
    for i in range(testing_data_length):
        if predicted_values[i]==actual_label[i]:
            count+=1
    print("Accuracy = ", (count/testing_data_length)*100, "%")

def classify_tree(predicted_values):
    # Train Data
    classify = tree.DecisionTreeClassifier()
    classify = classify.fit(training_data, training_label)
    
    predict_image(classify, predicted_values)
    calculate_accuracy(predicted_values)
    return predicted_values

def classify_svm():
    # Train Data
    classify = svm.SVC()
    classify = classify.fit(training_data, training_label)
    predicted_values = []

    predict_image(classify, predicted_values)
    calculate_accuracy(predicted_values)

def classify_KNeighbors():
    # Train Data
    classify = neighbors.KNeighborsClassifier()
    classify = classify.fit(training_data, training_label)
    predicted_values = []

    predict_image(classify, predicted_values)
    calculate_accuracy(predicted_values)

def classify_perceptron():
    # Train Data
    classify = linear_model.Perceptron()
    classify = classify.fit(training_data, training_label)
    predicted_values = []

    predict_image(classify, predicted_values)
    calculate_accuracy(predicted_values)

classify_tree(predicted_values)
#classify_svm()
#classify_KNeighbors()
#classify_perceptron()
#print(predicted_values)
#plot_images(predicted_values)