import numpy as np

from sklearn.model_selection import train_test_split
#from sklearn.model_selection import ShuffleSplit

#Feature selection libraries
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import statistics as st

import random

import matplotlib.pyplot as plt

#Image processing libraries
import cv2
import skimage.measure as sm

#Dataframes
import pandas as pd

#Machine learning libraries
import seaborn as sns; sns.set()

import sklearn.tree
import sklearn.neighbors

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

#Deep learning library : Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

################################################################
#Import labels
labels = pd.read_csv(r'C:\Users\Asus 6eme\Documents\MATLAB\TB3\Project (skin lesion classification)-20191220\ISIC-2017_Data_GroundTruth_Classification.csv')

#Loading Data 
k = 0
X_original = [] #Original images
X_r = [] #Their red component for the cnn
X_segmented = [] #Segmented images
X_superpixels = [] #Superpixels
indexes = []
Y = [] #Labels vector
for i in range(519):
    if (i < 10):
        name = r'C:\Users\Asus 6eme\Documents\MATLAB\TB3\Project (skin lesion classification)-20191220\PROJECT_Data\ISIC_000000' + str(i)
        I = cv2.imread(name + '.jpg')
        J = cv2.imread(name + '_segmentation.png', cv2.IMREAD_GRAYSCALE)
        S = cv2.imread(name + '_superpixels.png')
        N = 'ISIC_000000' + str(i)
    elif (i < 100):
        name = r'C:\Users\Asus 6eme\Documents\MATLAB\TB3\Project (skin lesion classification)-20191220\PROJECT_Data\ISIC_00000' + str(i)
        I = cv2.imread(name + '.jpg')
        J = cv2.imread(name + '_segmentation.png', cv2.IMREAD_GRAYSCALE)
        S = cv2.imread(name + '_superpixels.png')
        N = 'ISIC_00000' + str(i)
    else:
        name = r'C:\Users\Asus 6eme\Documents\MATLAB\TB3\Project (skin lesion classification)-20191220\PROJECT_Data\ISIC_0000' + str(i)
        I = cv2.imread(name + '.jpg')
        J = cv2.imread(name + '_segmentation.png', cv2.IMREAD_GRAYSCALE)
        S = cv2.imread(name + '_superpixels.png')
        N = 'ISIC_0000' + str(i)
    if (I is None):
        #in case of corresponding image of 'i' is missing
        print('fichier inexistant')
    else:
        #in case of corresponding image of 'i' is found
        k=1
    if (k == 1):
        indexes.append(i)
        k = 0
        print(i) #Useful for debeugguing
        
        X_original = X_original + [cv2.resize(src=I,dsize=(767,1022))]
        
        I_r = np.array(I[:,:,2])
        X_r = X_r + [cv2.resize(src=I_r,dsize=(112,84))]
        
        X_segmented = X_segmented + [255-J]
        
        X_superpixels = X_superpixels + [S]
        
        #Extract corresponding label
        for j in range(2000):
            if (labels['image_id'][j]==N):
                Y = Y + [int(labels['melanoma'][j])]; 
                break
#Classes names for better visualisations
class_names = ['benign','melanoma']

#A sample of original images labelled
fig=plt.figure()
for i in range(25):
    a=fig.add_subplot(5,5,i+1)
    a.set_title(class_names[Y[i]],color='blue')
    plt.imshow(cv2.resize(X_original[i], (102, 76)))
plt.show()

#tuvalum

#A sample of segmented images labelled
fig=plt.figure()
for i in range(25):
    a=fig.add_subplot(5,5,i+1)
    a.set_title(class_names[Y[i]],color='blue')
    plt.imshow(X_segmented[i],cmap='Greys')
plt.show()

#First approach : Region props on segmented images
train_images = X_segmented
train_labels = Y

#We will now describe our images geometrically using the region props function
imgdata=np.zeros((200,9))
print('Compute region props')
for i in range(200):
    #All poSsible properties were taken into consideration
    props = sm.regionprops_table(train_images[i], properties=['area', 'bbox_area','convex_area','eccentricity','equivalent_diameter','extent','major_axis_length','minor_axis_length','perimeter'])
    imgdata[i][0]=float(props['area'])
    imgdata[i][1]=float(props['bbox_area'])
    imgdata[i][2]=float(props['convex_area'])
    imgdata[i][3]=float(props['eccentricity'])
    imgdata[i][4]=float(props['equivalent_diameter'])
    imgdata[i][5]=float(props['extent'])
    imgdata[i][6]=float(props['major_axis_length'])
    imgdata[i][7]=float(props['minor_axis_length'])
    imgdata[i][8]=float(props['perimeter'])


#We will select only the 5 best features among the 9 computed
imgdata_new = SelectKBest(chi2, k=5).fit_transform(imgdata, train_labels)

#I won't go into details but the chi2 metric is adequat for the classification problem

#Superpixels processing
#Compute the number of superpixels in every image
numbre_sp = []
for i in range(200):
    print('superpixels image nmr :', i)
    sp = X_superpixels[i]
    
    #The red component
    r = np.array(sp[:,:,2])
    r = np.reshape(r, (1,-1))
    r = r[0]
    
    #the green component
    g = np.array(sp[:,:,1])
    g = np.reshape(g, (1,-1))
    g = g[0]
    
    #indexes of the last layer which are significative
    indexes = [i for i in range(len(g)) if g[i] == 3]
    
    #extraction of the superpixels in this layer
    r_bis = r[indexes]
    
    #Removing repeated values
    r_sub = list(dict.fromkeys(r[indexes]))
    
    numbre_sp = numbre_sp + [max(r_sub)]

numbre_sp = np.array(numbre_sp)
numbre_sp = np.reshape(numbre_sp, (200,1))

all_features = np.hstack((imgdata_new,numbre_sp)) #adding the number of superpixels to the features matrix
all_features_c = np.hstack((imgdata,numbre_sp))

#Convert data into DataFrame
df = pd.DataFrame(all_features, columns=['feature1', 'feature2','feature3','feature4','feature5','superpixels'])
df_c = pd.DataFrame(all_features_c, columns=['feature1', 'feature2','feature3','feature4','feature5', 'feature6','feature7','feature8','feature9','superpixels'])

print('start')

# Defining the machine learning models
logistic_m = LogisticRegression() #Logistic regression
tree_m = sklearn.tree.DecisionTreeClassifier(max_depth=3) #Decision tree
gradient_descent_m=SGDClassifier()# stochastic gradient descent
gradient_boosting_m=GradientBoostingClassifier()# gradient boosting
knn_m = sklearn.neighbors.KNeighborsClassifier(n_neighbors=13) #k nearest neighbors
gnb_m = GaussianNB() #Naive bayesian 

print('learn')

#Cross validation
n_samples = 100
accuracy1 = np.zeros(n_samples)
accuracy2 = np.zeros(n_samples)
accuracy3 = np.zeros(n_samples)
accuracy4 = np.zeros(n_samples)
accuracy5 = np.zeros(n_samples)
accuracy6 = np.zeros(n_samples)
for k in range(n_samples):
    print('sample : ', k)
    #splitting data into training and test sets
    random.seed(k)
    X_train, X_test, y_train, y_test = train_test_split(df_c, train_labels, test_size = 0.25)
    
    #Train models with the method .fit()
    model1 = logistic_m.fit(X_train, y_train)
    print('1')
    model2 = tree_m.fit(X_train, y_train)
    print('2')
    model3 = gradient_descent_m.fit(X_train, y_train)
    print('3')
    model4 = gradient_boosting_m.fit(X_train, y_train)
    print('4')
    model5 = knn_m.fit(X_train, y_train)
    print('5')
    model6 = gnb_m.fit(X_train, y_train)
    
    print('predict')
    
    # Predictions are simply computed by the method .predict()
    predictions1 = logistic_m.predict(X_test)
    predictions2 = tree_m.predict(X_test)
    predictions3 = gradient_descent_m.predict(X_test)
    predictions4 = gradient_boosting_m.predict(X_test)
    predictions5 = knn_m.predict(X_test)
    predictions6 = gnb_m.predict(X_test)
    
    print('evaluate')
    
    #Compute accuracy for each model
    accuracy1[k] = accuracy_score(y_test,predictions1)
    accuracy2[k] = accuracy_score(y_test,predictions2)
    accuracy3[k] = accuracy_score(y_test,predictions3)
    accuracy4[k] = accuracy_score(y_test,predictions4)
    accuracy5[k] = accuracy_score(y_test,predictions5)
    accuracy6[k] = accuracy_score(y_test,predictions6)

#Average accuracy of each model   
accuracy1_m = st.mean(accuracy1)
accuracy2_m = st.mean(accuracy2)
accuracy3_m = st.mean(accuracy3)
accuracy4_m = st.mean(accuracy4)
accuracy5_m = st.mean(accuracy5)
accuracy6_m = st.mean(accuracy6)

#Results
print(['Logitic regression',accuracy1_m])
print(['Decision tree',accuracy2_m])
print(['Gradient descent',accuracy3_m])
print(['Gradient boosting',accuracy4_m])
print(['Knn',accuracy5_m])
print(['Naive Bayesian',accuracy6_m])

#Plot of accuracy
fig=plt.figure()
plt.bar([1,2,3,4,5,6],height=[accuracy1_m,accuracy2_m,accuracy3_m,accuracy4_m,accuracy5_m,accuracy6_m],tick_label=['Logitic \n regression', 'Decision \n tree', 'Gradient \n descent', 'Gradient \n boosting', 'Knn','Naive bayesian'])
plt.title('models accuracy')
plt.ylabel('accuracy')
plt.xlabel('model')
plt.show()

#visualisation of a sample of the predicted data from the best model with their labels
indexes = X_test.index
fig=plt.figure()
for i in range(25):
    a=fig.add_subplot(5,5,i+1)
    if predictions4[i] == y_test[i]:
        col='blue'
    else:
        col= 'red'
    a.set_title(class_names[predictions5[i]],color=col)
    plt.imshow(train_images[indexes[i]],cmap='Greys')
plt.show()

###############################################################################

#Deep learning classifier using tensorflow
train_images = X_r
train_labels = Y

# Splitting data into training and test data
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size = 0.25)

#Again into validation set
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.25)

X_train = np.array(X_train)
X_test = np.array(X_test)
X_validation = np.array(X_validation)

#reshaping for adequacy with the cnn layers
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

#Verification
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('X_validation shape: ', X_validation.shape)

batch_size = 32 # You can try 64 or 128 if you'd like to
num_classes = 2
epochs = 25 # loss function value will be stabilized after 93rd epoch# To save the model:
lr = 0.001

input_shape = (84, 112, 1)

#Sequential to concatenate the layers we are willing to add
model = Sequential()

#Three convolutional layers with respectively 8, 16 and 32 neurons
model.add(Conv2D(8, (3, 3), padding="same", kernel_regularizer=l2(0.01), input_shape=input_shape))
#model.add(Conv2D(16, (3, 3), padding="same", kernel_regularizer=l2(0.01), input_shape=input_shape))
model.add(Conv2D(32, (3, 3), padding="same", kernel_regularizer=l2(0.01), input_shape=input_shape))
model.add(Activation("relu"))

#Batch normalization is added to limit changes in the distribution of the input values in a machine
# learning algorithm (Covariate shift).
model.add(BatchNormalization(axis=-1))

#reducing dimension by selecting the Max in each 2*2 window
model.add(MaxPooling2D(pool_size=(2, 2)))
#Dropout take off 15/100 of output values from the last layer in order to avoid overfitting
model.add(Dropout(0.15))

#flatten the layers
model.add(Flatten())

#Adding a normal Dense layer with 16 neurones
model.add(Dense(16))
model.add(Activation("relu"))
model.add(Dropout(0.25))

#And finally a normal dense layer with number of outputs the same as the number of classes we have
model.add(Dense(num_classes))
model.add(Activation("softmax"))

#Parametrizing the optimizer of the gradient descent
opt = Adam(lr=lr)

#Compiling the model specifying the metrics we want to check
model.compile(loss='categorical_crossentropy', 
              optimizer=opt,
              metrics=['accuracy'])

y_train_c = to_categorical(y_train, num_classes=2, dtype='float32')


print('start learning')
history=model.fit(X_train.astype("float32"), y_train_c,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test.astype("float32"), to_categorical(y_test)),
              shuffle=True)

print(history.history)
scores = model.evaluate(X_validation.astype("float32"), to_categorical(y_validation), verbose=0)
print('Validation loss:', scores[0])
print('Validation accuracy:', scores[1])

print(history.history.keys())


fig=plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Test on the validation Data
predictions = model.predict(X_validation.astype("float32"))
               
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2])

#Visualize results
fig=plt.figure()
for i in range(17):
    a=fig.add_subplot(5,5,i+1)
    t=predictions[i,:]
    
    if np.argmax(t) == y_validation[i]:
        col='blue'
    else:
        col= 'red'
    
    a.set_title(class_names[np.argmax(t)],color=col)
    plt.imshow(X_validation[i],cmap='Greys')
plt.show()

#Sophisticated visualization with percentage of belonging to each class
fig=plt.figure()
k=0
for i in range(10,16):
    plt.subplot(3, 4, 2*k+1)
    plt.imshow(X_validation[i],cmap='Greys')
    
    t=predictions[i,:]
    if np.argmax(t) == y_validation[i]:
        col='blue'
    else:
        col= 'red'
    
    plt.xlabel(class_names[np.argmax(t)]+' '+str(int(100*np.max(t)))+'% ('+class_names[y_validation[i]]+')',color=col)  
       
    plt.subplot(3, 4, 2*k+2)
    bar = plt.bar(range(2), t, color="#777777")
    
    bar[np.argmax(t)].set_color('red')
    bar[y_validation[i]].set_color('blue')
    k=k+1
plt.tight_layout()
plt.show()
