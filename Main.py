from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import ttk
from tkinter import filedialog
from tkinter import END  
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import joblib
from scipy.stats import mode
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import roc_curve

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

from sklearn.ensemble import VotingClassifier
from keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.preprocessing import normalize, LabelEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import  ExtraTreesClassifier, VotingClassifier,RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

main = Tk()
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy
global accuracy, precision, recall, f1
from tensorflow.keras.applications.vgg19 import VGG19

# Initialize empty lists for features and labelsz
X = []
Y = []

base_model = VGG19(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
model_folder = "model"

def uploadDataset():
    global filename,categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")

def VGG19_feature_extraction(): 
    global X, Y, base_model, categories, filename
    text.delete('1.0', END)

    model_data_path = "model/X_compressed.npz"
    model_label_path = "model/Y.npy"

    if os.path.exists(model_data_path) and os.path.exists(model_label_path):
        X = np.load(model_data_path)['X']  # Load from compressed file
        Y = np.load(model_label_path)
    else:
        X = []
        Y = []
        data_folder = filename
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

        for class_label, class_name in enumerate(categories):
            class_folder = os.path.join(data_folder, class_name)
            for img_file in os.listdir(class_folder):
                if img_file.lower().endswith(valid_extensions):
                    img_path = os.path.join(class_folder, img_file)
                    print(img_path)
                    img = image.load_img(img_path, target_size=(224,224))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = base_model.predict(x, verbose=0)
                    features = np.squeeze(features)  # Flatten features
                    X.append(features)
                    Y.append(class_label)
        
        # Convert lists to NumPy arrays
        X = np.array(X)
        Y = np.array(Y)

        # Save processed images and labels (compressed format for X)
        np.savez_compressed(model_data_path, X=X)
        np.save(model_label_path, Y)
            
    text.insert(END, "Image Preprocessing Completed\n")
    text.insert(END, "VGG19 Feature Extraction Completed\n")
    text.insert(END, f"Feature Dimension: {X.shape}\n")


  
def Train_test_spliting():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    
    
    X_downsampled = X
    Y_downsampled = Y
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X_downsampled = X_downsampled[indices]
        Y_downsampled = Y_downsampled[indices]  
    else:
        indices = np.arange(X_downsampled.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X_downsampled = X_downsampled[indices]
        Y_downsampled = Y_downsampled[indices]
        
    
    X_train, X_test, y_train, y_test = train_test_split(X_downsampled, Y_downsampled, test_size=0.2, random_state=42)

    text.insert(END, f"Input Data Train  Size: {X_train.shape}\n")
    text.insert(END, f"Input Data Test  Size: {X_test.shape}\n")
    text.insert(END, f"Output  Train Size: {y_train.shape}\n")
    text.insert(END, f"Output  Test Size: {y_test.shape}\n")

    


def performance_evaluation(label,algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()     

    
    return (a, p, r, f)



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import joblib
import os

def Model_LDA():
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)
    model_filename = os.path.join(model_folder, "LDA_model.pkl")

    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = LinearDiscriminantAnalysis()
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)

    Y_pred = Model1.predict(X_test)
    performance_evaluation(categories, "Existing LDA", y_test, Y_pred)



def Model_QDA():
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)
    model_filename = os.path.join(model_folder, "QDA_model.pkl")

    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = QuadraticDiscriminantAnalysis()
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)

    Y_pred = Model1.predict(X_test)
    performance_evaluation(categories, "Existing QDA", y_test, Y_pred)




from sklearn.svm import SVC
def Model_SVM():
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)
    model_filename = os.path.join(model_folder, "SVM_model.pkl")

    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        # stronger SVM hyperparameters
        Model1 = SVC(
            kernel='rbf',
            C=100,            # higher C for narrower margin
            gamma=0.001,      # smaller gamma for smoother decision boundary
            class_weight='balanced',
            probability=True
        )
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename, compress=3)

    Y_pred = Model1.predict(X_test)
    performance_evaluation(categories, "Existing SVM Classifier", y_test, Y_pred)



from sklearn.ensemble import ExtraTreesClassifier

def ETC_model():
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)
    model_filename = os.path.join(model_folder, "ETC_model.pkl")
    
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = ExtraTreesClassifier(
            n_estimators=200,        # Number of trees
            max_depth=32,          # Expand nodes until all leaves are pure
            min_samples_split=2,     # Minimum samples to split an internal node
            min_samples_leaf=1,      # Minimum samples required at each leaf node
            max_features="sqrt",     # Number of features considered for splits
            bootstrap=False,         # Whether bootstrap samples are used
           
            random_state=42          # Reproducibility
        )
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)
    
    Y_pred = Model1.predict(X_test)
    performance_evaluation(categories, "Proposed VGG19 with ETC", y_test, Y_pred)


def predict():
    global  base_model,categories,optimal_model,Model1

    categories=['good', 'hole', 'objects']
    model_filename = os.path.join(model_folder, "ETC_model.pkl")
    
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        print('got error')

    filename = filedialog.askopenfilename(initialdir="testImages")
    img_path = filename
    img = image.load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)    
    features = base_model.predict(x)
    preds  = Model1.predict(features.reshape(1, -1))
    
    if isinstance(preds, (list, np.ndarray)):
        preds = int(preds[0])  # Adjust this line based on the structure of your preds array
    else:
        preds = int(preds)
           
    # Display the result on the image
    img = cv2.imread(filename)
    img = cv2.resize(img, (800, 800))
    
    class_label = categories[preds]

    text_to_display = f'Output Classified as: {class_label}'
    cv2.putText(img, text_to_display,  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(f'Output Classified as: {class_label}',img)
    cv2.waitKey(0)

def close():
    main.destroy()
    #text.delete('1.0', END)



import os
import joblib
import tkinter as tk
from tkinter import filedialog, Text, ttk
from PIL import Image, ImageTk

# Load Background Image
bg_image = Image.open("background.jpg")  # Change with your image file
bg_image = bg_image.resize((1380, 730), Image.LANCZOS)  # Resize to match window
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(main, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)  # Cover entire window

    
font = ('times', 15, 'bold')
title = Label(main, text='Patch-Level Texture Classification in Remote Sensing Using Transfer Learning on TILDA-400')
#title.config(bg='LightBlue1', fg='black')  
title.config(font=font)           
#title.config(height=3, width=120)       
title.place(x=300,y=15)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset, bg="lightblue")
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)

processButton = Button(main, text="VGG19 Feature extraction", command=VGG19_feature_extraction, bg="lightblue")
processButton.place(x=20,y=150)
processButton.config(font=ff)

processButton = Button(main, text="Train Test Splitting", command=Train_test_spliting, bg="lightblue")
processButton.place(x=20,y=200)
processButton.config(font=ff)


modelButton = Button(main, text="Existing LDA", command=Model_LDA, bg="lightblue")
modelButton.place(x=20,y=250)
modelButton.config(font=ff)

modelButton = Button(main, text="Existing QDA", command=Model_QDA, bg="lightblue")
modelButton.place(x=20,y=300)
modelButton.config(font=ff)

modelButton = Button(main, text="Train SVM", command=Model_SVM, bg="lightblue")
modelButton.place(x=20,y=350)
modelButton.config(font=ff)

modelButton = Button(main, text="Train ETC", command=ETC_model, bg="lightblue")
modelButton.place(x=20,y=400)
modelButton.config(font=ff)

predictButton = Button(main, text="Upload test image", command=predict, bg="lightblue")
predictButton.place(x=20,y=450)
predictButton.config(font=ff)


exitButton = Button(main, text="Exit", command=close, bg="lightblue")
exitButton.place(x=20,y=500)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=70)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config(bg='SkyBlue')
main.mainloop()