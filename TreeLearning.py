#import numpy as np
#import cv2
#import mahotas
##funcao para facilitar a escrita nas imagem
#def escreve(img, texto, cor=(255,0,0)):
#    cv2.putText(img, texto, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 0, cv2.LINE_AA)
#imgcolorida = cv2.imread('1555160533023_2.jpeg') #carregamento da imagem
#imgcolorida = cv2.resize(imgcolorida, (600,600))
##se necessario o redimensioamento da imagem pode vir aqui.
##passo 1: conversao para tons de cinza
#img = cv2.cvtColor(imgcolorida, cv2.COLOR_BGR2GRAY)
##passo 2: blur/suavizacao da imagem
#suave = cv2.blur(img, (11, 11), 0)
##suave = cv2.medianblur(img,35)
##passo 3: binarizacao resultando em pixels brancos e pretos
#t = mahotas.thresholding.otsu(suave)
#bin = suave.copy()
#bin[bin > t] = 255
#bin[bin < 255] = 0
#bin = cv2.bitwise_not(bin)
##passo 4: deteccao de bordas com canny
#bordas = cv2.Canny(bin, 70, 150)
#cv2.imshow("teste", bordas)
#cv2.waitKey(0)
##passo 5: identificacao e contagem dos contornos da imagem
##cv2.retr_external = conta apenas os contornos externos
#(objetos, lx) = cv2.findContours(bordas.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##a variavel lx (lixo) recebe dados que nao sao utilizados
#escreve(img, "imagem em tons de cinza", 0)
#escreve(suave, "suavizacao com blur", 0)
#escreve(bin, "binarizacao com metodo otsu", 255)
#escreve(bordas, "detector de bordas canny", 255)
#temp = np.vstack([
#np.hstack([img, suave]),
#np.hstack([bin, bordas])
#])
#cv2.imshow("quantidade de objetos: "+str(len(objetos)), temp)
#cv2.waitKey(0)
#imgc2 = imgcolorida.copy()
#cv2.imshow("imagem original", imgcolorida)
#cv2.drawContours(imgc2, objetos, -1, (255, 0, 0), 2)
#escreve(imgc2, str(len(objetos))+" objetos encontrados!")
#cv2.imshow("resultado", imgc2)
#cv2.waitKey(0)

#import cv2
#import matplotlib.pyplot as plt
#import numpy as np

#flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
#len(flags)
#flags[40]
#nemo = cv2.imread('teste.jpeg')
#nemo = cv2.resize(nemo, (600,600))
#plt.imshow(nemo)
#plt.show()
#nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
#plt.imshow(nemo)
#plt.show()
#hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
##light_orange = (234, 240, 226)
##dark_orange = (255, 255, 255)
##mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
##result = cv2.bitwise_and(nemo, nemo, mask=mask)
##plt.subplot(1, 2, 1)
##plt.imshow(mask, cmap="gray")
##plt.subplot(1, 2, 2)
##plt.imshow(result)
##plt.show()
#light_white = (108, 98,105)
#dark_white = (72,65,57)
#plt.show(dark_white)
#mask_white = cv2.inRange(nemo, light_white, dark_white)
#result_white = cv2.bitwise_and(nemo, nemo, mask=mask_white)

#plt.subplot(1, 2, 1)
#plt.imshow(mask_white, cmap="gray")
#plt.subplot(1, 2, 2)
#plt.imshow(result_white)
#plt.show()
##final_mask = mask + mask_white
#final_mask = mask_white

#final_result = cv2.bitwise_and(nemo, nemo, mask=final_mask)
#plt.subplot(1, 2, 1)
#plt.imshow(final_mask, cmap="gray")
#plt.subplot(1, 2, 2)
#plt.imshow(final_result)
#plt.show()
#blur = cv2.GaussianBlur(final_result, (7, 7), 0)
#plt.imshow(blur)
#plt.show()

#import numpy as np
#import cv2
#from matplotlib import pyplot as plt
#import mahotas
#img = cv2.imread('teste.jpeg')
#img = cv2.resize(img, (600,600),interpolation = cv2.INTER_AREA)
#mask = np.zeros(img.shape[:2],np.uint8)
#bgdModel = np.zeros((1,65),np.float64)
#fgdModel = np.zeros((1,65),np.float64)
##x,y,w,h
#rect = (200,0,300,560)
#cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#img = img*mask2[:,:,np.newaxis]
#plt.imshow(img),plt.colorbar(),plt.show()

##funcao para facilitar a escrita nas imagem
#def escreve(img, texto, cor=(255,0,0)):
#    cv2.putText(img, texto, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 0, cv2.LINE_AA)
#imgcolorida = cv2.imread('1555160533023_2.jpeg') 
##carregamento da imagem
##imgcolorida = cv2.resize(imgcolorida, (600,600))
##se necessario o redimensioamento da imagem pode vir aqui.
##passo 1: conversao para tons de cinza
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##passo 2: blur/suavizacao da imagem
#suave = cv2.blur(img, (11, 11), 0)
##suave = cv2.medianblur(img,35)
##passo 3: binarizacao resultando em pixels brancos e pretos
#t = mahotas.thresholding.otsu(suave)
#bin = suave.copy()
#bin[bin > t] = 255
#bin[bin < 255] = 0
#bin = cv2.bitwise_not(bin)
##passo 4: deteccao de bordas com canny
#bordas = cv2.Canny(bin, 70, 150)
#cv2.imshow("teste", bordas)
#cv2.waitKey(0)
##passo 5: identificacao e contagem dos contornos da imagem
##cv2.retr_external = conta apenas os contornos externos
#(objetos, lx) = cv2.findContours(bordas.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##a variavel lx (lixo) recebe dados que nao sao utilizados
#escreve(img, "imagem em tons de cinza", 0)
#escreve(suave, "suavizacao com blur", 0)
#escreve(bin, "binarizacao com metodo otsu", 255)
#escreve(bordas, "detector de bordas canny", 255)
#temp = np.vstack([
#np.hstack([img, suave]),
#np.hstack([bin, bordas])
#])
#cv2.imshow("quantidade de objetos: "+str(len(objetos)), temp)
#cv2.waitKey(0)
#imgc2 = img.copy()
#cv2.imshow("imagem original", img)
#cv2.drawContours(imgc2, objetos, -1, (255, 0, 0), 2)
#escreve(imgc2, str(len(objetos))+" objetos encontrados!")
#cv2.imshow("resultado", imgc2)
#cv2.waitKey(0)

import cv2
import numpy as np
from matplotlib import pyplot as plt
import mahotas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

#X = []
#N=3
#for i in range(N):
#    im = cv2.imread('teste.jpeg')
#    X.append(im[0:100,0:100])
#    cv2.imshow("resultado crop", X[0])
#    cv2.waitKey(0)

#import numpy as np
#import cv2
#y=0
#x=0
#N=6
#X=[]
#h=100
#w=100
#image = cv2.imread('teste.jpeg')
#img = cv2.resize(image, (600,600),interpolation = cv2.INTER_AREA)
#for i in range(N):
#    x=100*i
#    for j in range(N):
#        y=100*j         
#        crop = img[y:y+h, x:x+w]
#        X.append(crop)
#        #cv2.imshow('Image', img)
#        #cv2.waitKey(0) 
#        #cv2.imshow('Image', X[j+(5*i)])
#        #cv2.waitKey(0)       
#color = ('b','g','r')
#for imagem in X:
#    cv2.imshow('Image', imagem)
#    for i,col in enumerate(color):
#        histr = cv2.calcHist([imagem],[i],None,[256],[0,256])
#        plt.plot(histr,color = col)
#        plt.xlim([0,256])
#    # convert the image to grayscale
#    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
#    # compute the haralick texture feature vector
#    haralick = mahotas.features.haralick(gray).mean(axis=0)
#    # return the result
#    print haralick
#    plt.show()
#    cv2.waitKey(0)

import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
#name=0
#for file in glob.glob( "./Fotos/Fotos/*.jpeg"):    
#    y=0
#    x=0
#    N=6
#    h=100
#    w=100
#    X=[]
#    image = cv2.imread(file)
#    img = cv2.resize(image, (600,600),interpolation = cv2.INTER_AREA)
#    subname=0
#    for i in range(N):
#        x=100*i
#        for j in range(N):
#            y=100*j         
#            crop = img[y:y+h, x:x+w]            
#            cv2.imwrite("./Fotos/Fotos/Dataset/"+str(name)+"_"+str(subname)+".jpeg",crop)
#            X.append(crop)
#            subname=subname+1
#    name=name+1

# function to extract haralick textures from an image
def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    #textures = mt.features.haralick(image)

    ## take the mean of it and return it
    #ht_mean  = textures.mean(axis=0)
    #return ht_mean

    color = ('b','g','r')

    #cv2.imshow('image', image)
    for i,col in enumerate(color):
        histr = cv2.calcHist([image],[1],None,[256],[0,256])
                
    #    plt.plot(histr,color = col)
    #    plt.xlim([0,256])
        hist= np.concatenate(histr)

    return hist

# load the training dataset
train_path  = "./Fotos/Fotos/Dataset2/train/"
train_names = os.listdir(train_path)
print train_names

# empty list to hold feature vectors and train labels
train_features = []
train_labels   = []

# loop over the training dataset
print "[STATUS] Started extracting features.."
for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        i = 1
        for file in glob.glob(cur_path + "/*.jpeg"):
                #print "Processing Image - {} in {}".format(i, cur_label)
                # read the training image
                image = cv2.imread(file)

               

                # convert the image to grayscale
                #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # extract haralick texture from the image
                features = extract_features(image)
                #print features

                # append the feature vector and label
                train_features.append(features)
                #print features
                train_labels.append(cur_label)

                # show loop update
                i += 1
# have a look at the size of our feature vector and labels
print "Training features: {}".format(np.array(train_features).shape)
print "Training labels: {}".format(np.array(train_labels).shape)


# create the classifier
print "[STATUS] Creating the classifier.."
#clf_svm = LinearSVC(random_state=20)
#lda = LinearDiscriminantAnalysis(n_components=4)
#train_features_lda = lda.fit_transform(train_features, train_labels)
modelN= KNeighborsClassifier()
#print train_features[0]

# fit the training data and labels
print "[STATUS] Fitting data/label to model.."
#clf_svm.fit(train_features, train_labels)
#lda=model.fit(train_features, train_labels).transform(train_features)
modelN.fit(train_features, train_labels)

# loop over the test images

for file in glob.glob("./Fotos/Fotos/Dataset2/*.jpeg"):
    # read the input image
    image = cv2.imread(file)

    # convert to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # extract haralick texture from the image
    features = extract_features(image)
    #features_lda = lda.transform(features.reshape(1,-1))[0]
    #lda2=model.fit(features, train_labels).transform(features)

	# evaluate the model and predict label
    prediction = modelN.predict(features.reshape(1, -1))[0]

	# show the label
    cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    print "Prediction - {}".format(prediction)

	# display the output image
    cv2.imshow("Test_Image", image)
    cv2.waitKey(0)     