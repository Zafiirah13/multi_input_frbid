#!usr/bin/env python
"""
Authors : Zafiirah Hosenie
Email : zafiirah.hosenie@gmail.com or zafiirah.hosenie@postgrad.manchester.ac.uk
Affiliation : The University of Manchester, UK.
License : MIT
Status : Under Development
Description :
Python implementation for FRBID: Fast Radio Burst Intelligent Distinguisher.
Based on Multi-Input Convolutional Neural Network
This code is tested in Python 3 version 3.5.3  
"""

import keras
from keras.models import Sequential,Model
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation, Conv2D, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU


def create_convolution_layers(input_img, img_shape):
    '''
    This function create the network for each input

    Params:
        input_img  : nput(shape=img_shape)
        input_shape: The shape of the input image for e.g (256, 256, 1), 
                        or (X_train_dm.shape[1],X_train_dm.shape[2],X_train_dm.shape[3]) 
                        or (X_train_freq.shape[1],X_train_freq.shape[2],X_train_freq.shape[3])
    '''
    A = 16; B = 32; C = 64
    #A = 512; B = 1024; C = 2048

    model = Conv2D(A, (3, 3),activation='relu', input_shape=img_shape)(input_img)
    model = MaxPooling2D((2, 2))(model)
    model = Dropout(0.1)(model)

    model = Conv2D(B, (3, 3),activation='relu')(model)
    model = MaxPooling2D((2, 2))(model)
    model = Dropout(0.1)(model)

    model = Conv2D(C, (3, 3),activation='relu')(model)
    model = MaxPooling2D((2, 2))(model)
    model = Dropout(0.1)(model)

    return model



def get_modelparameters(params,dm_img_shape,freq_img_shape, lr):
    '''
    This function calls out the model we want for training the images

    INPUT
        params: The model name we want to train for e.g 'MULTIINPUT'
        img_shape: The shape of the image or (256, 256, 1), or (X_train_dm.shape[1],X_train_dm.shape[2],X_train_dm.shape[3]) or (X_train_freq.shape[1],X_train_freq.shape[2],X_train_freq.shape[3])
        lr: The learning rate for the optimisation values can vary from [0.1, 0.01, 0.001, 0.0001]
    '''

    if params == 'MULTIINPUT':

        # initialise the dm input and pass it through the network
        dm_input = Input(shape=dm_img_shape)
        dm_model = create_convolution_layers(dm_input,dm_img_shape)

        # Initialise the freq input and pass it through the network
        freq_input = Input(shape=freq_img_shape)
        freq_model = create_convolution_layers(freq_input,freq_img_shape)

        # concatenate both model- dm and frequency model
        conv = concatenate([dm_model, freq_model])

        # Flatten the weights
        conv = Flatten()(conv)

        # Pass the concatenate model through one last layers of dense layers
        dense = Dense(512)(conv)
        dense = Activation('relu')(dense)
        dense = Dropout(0.5)(dense)

        dense = Dense(1024)(dense)
        dense = Activation('relu')(dense)
        dense = Dropout(0.5)(dense)

        dense = Dense(2048)(dense)
        dense = Activation('relu')(dense)
        dense = Dropout(0.5)(dense)

        output = Dense(2, activation='softmax')(dense)

        model = Model(inputs=[dm_input, freq_input], outputs=[output])

        model.summary()

        opt = keras.optimizers.Adam(lr)

        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])


    return model



def compile_model(params,dm_img_shape,freq_img_shape,save_model_dir, X_train_dm,X_train_freq, 
                y_train, X_val_dm,X_val_freq, y_val, batch_size, epochs, lr, class_weight, 
                early_stopping=False,save_model=False,data_augmentation=False):
    '''
    This function compile the model, apply early stopping to avoid overfitting and also apply data augmentation
    if we set it to True

    INPUTS:
        params: The model name we want to train for e.g 'MULTIINPUT'
        dm_img_shape: The shape of the dm image (256,256,1), or (X_train_dm.shape[1],X_train_dm.shape[2],X_train_dm.shape[3])
        freq_img_shape: The shape of the freq image (256,256,1), or (X_train_freq.shape[1],X_train_freq.shape[2],X_train_freq.shape[3])
        save_model_dir: The directory we want to save the history of the model [accuracy, loss]
        X_train_dm, X_train_freq: The training set for dm and freq images, say having shape (Nimages, 256pix, 256pix, 1 channel)
        X_val_dm,X_val_freq: The tvalidation set for dm and freq images, say having shape (Nimages, 256pix, 256pix, 1 channel)
        y_train, y_val: The label for training and validation set- transform to one-hot encoding having shape (Nimages, 2) in the format array([[0., 1.],[1., 0.])
        batch_size: Integer values values can be in the range [32, 64, 128, 256]
        epoch: The number of iteration to train the network. Integer value varies in the range [10, 50, 100, 200, ...]
        lr: The learning rate for the optimisation values can vary from [0.1, 0.01, 0.001, 0.0001]
        class_weight: If we want the model to give more weights to the class we are interested then set it to {0:0.25,1:0.75} or None
        early_stopping: Stop the network from training if val_loss stop decreasing if TRUE
        save_model: set TRUE to save the model after training
        data_augmentation: set TRUE if we want to apply data augmentation

    OUTPUTS:
        history: The logs of the accuracy and loss during optimization
        modelCNN: The fully trained model

    ''' 
    if save_model:
        tensorboard = TensorBoard(log_dir = save_model_dir + 'logs', write_graph=True)
 
    callbacks = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
    modelCNN  = get_modelparameters(params,dm_img_shape,freq_img_shape, lr)



    if not data_augmentation:
        print('Not using data augmentation.')
        if early_stopping:
            history       = modelCNN.fit([X_train_dm, X_train_freq], y_train,
                                         batch_size, 
                                         epochs, 
                                         validation_data=([X_val_dm, X_val_freq],y_val),
                                         class_weight = class_weight,
                                         verbose=1,
                                         callbacks=[callbacks],
                                         shuffle=True)
        else:
            history = modelCNN.fit([X_train_dm, X_train_freq], y_train,
                                    batch_size, 
                                    epochs, 
                                    validation_data=([X_val_dm, X_val_freq],y_val),
                                    class_weight = class_weight,
                                    verbose=1,
                                    shuffle=True)

    else:
        print('Using real-time data augmentation.')
        aug = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,fill_mode="nearest")

        if early_stopping:
            history       = modelCNN.fit_generator(aug.flow([X_train_dm, X_train_freq], y_train,
                                                    batch_size=batch_size),
                                                    steps_per_epoch=len(X_train_dm) // batch_size,
                                                    epochs=epochs,
                                                    validation_data=([X_val_dm, X_val_freq],y_val),
                                                    class_weight = class_weight,
                                                    verbose=1,
                                                    callbacks=[callbacks],
                                                    shuffle=True)

        else:
            history = modelCNN.fit_generator(aug.flow([X_train_dm, X_train_freq], y_train,
                                             batch_size=batch_size),
                                             steps_per_epoch=len(X_train_dm) // batch_size,
                                             epochs=epochs,
                                             validation_data=([X_val_dm, X_val_freq],y_val),
                                             class_weight = class_weight,
                                             verbose=1,
                                             shuffle=True)
    return history, modelCNN




def model_save(model, model_name):
    '''
    Function to save the fully trained model

    INPUTS:
        model: Here it will be modelCNN, that is the fully trained network
        model_name: the name of the fully trained network
    '''
    # serialize model to JSON
    model_json = model.to_json()
    with open('./frbid_model/'+model_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights('./frbid_model/'+model_name+".h5")
    print("Saved model to disk")            
    return model


