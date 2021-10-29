# Rafael Espinosa Mena
# rafaelespinosa4158@gmail.com
# Handwritten Latin Letters Classifier using MLPClassifier

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import random
from sklearn.neural_network import MLPClassifier

# Read data into dataset
hand_df = pd.read_csv("A_Z Handwritten Data.csv")
print(hand_df)


# Separate dataframe into features and target set
X = hand_df.iloc[:,1:] # Don't include the first column
Y = hand_df.iloc[:,0]

# Print shape of feature and target sets
print(X.shape)
print(Y.shape)

# Map numbers to letters
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',
             11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',
             20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

# Show a histogram of the letters
Y_1 = Y.map(word_dict)
plt.hist(Y_1, bins=26)
plt.show()

# Display one random letter from the dataset
r = random.randint(0, hand_df.shape[0])
pixel = hand_df.iloc[r,1:] # don't include label
# Shape pixel as a 28x28 array of greyscale
pixel = np.array(pixel)
pixel = pixel.reshape(28,28)
plt.imshow(pixel, cmap="gray")
letterIs = word_dict[hand_df.iloc[r, 0]]
plt.title("The letter is " + str(letterIs))
plt.show()

# Partition data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y, test_size=0.3, random_state=2021, stratify=Y)


# Scale and train the test features
X_train = X_train/255
X_test = X_test/255

# Crate MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), activation="relu",
                    max_iter=12, alpha=0.0001, solver="adam",
                    random_state=2021, learning_rate_init=0.01, verbose=True)

# Fit the data to the model
mlp.fit(X_train, Y_train)

# plot the loss curve
plt.plot(mlp.loss_curve_)
plt.show()

# Display accuracy of your model
print("The accuracy of the test data is", mlp.score(X_test, Y_test))

# 15. Display confusion matrix
from sklearn.metrics import plot_confusion_matrix
y_pred = mlp.predict(X_test)
cm = confusion_matrix(y_pred, Y_test)
plot_confusion_matrix(mlp, X_test, Y_test)
plt.show()
print("The confusion matrix", cm)
print("The predicted letter is " + str(y_pred[0]))

# 1Display predicted vs. actual letter for first letter in test set
test_sample = np.array(X_test.iloc[0]).reshape(28,28)
plt.imshow(test_sample, cmap="gray")
Y_testlist = Y_test.tolist()
plt.title("The predicted letter is " + word_dict[y_pred[0]]
          + ". The human tagged label is " + word_dict[Y_testlist[0]])
plt.show()


# Display a misclassified letter
# Show misclassified digit as an image. And show the predicted and actual
# digits as the title of that image.
failed_df = X_test[y_pred != Y_test]
# print(failed_df)
# Now pick a random row from the failed data frame
failed_index = failed_df.sample(n=1).index
failed_index1 = failed_index.tolist()[0]
# reshape into 28x28
failed_sample = np.array(X_test.iloc[failed_index]).reshape(28,28)
# plot the digit as an image
plt.imshow(failed_sample, cmap="gray")
# display the predicted and actual labels
plt.title("The predicted letter is " + word_dict[y_pred[failed_index1]]
          + ". The human tagged label is " + word_dict[Y_test[failed_index1]])
# show the plot
plt.show()



