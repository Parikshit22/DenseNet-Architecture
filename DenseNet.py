# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:43:38 2019

@author: MUJ
"""

import keras
from keras.models import Model
from keras.layers import Dense,MaxPooling2D,Conv2D,AveragePooling2D,BatchNormalization,Input,Add,ZeroPadding2D,Activation,GlobalAveragePooling2D,Flatten
def Dense_Layer(x,k):
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = Conv2D(4*k,(1,1),strides = (1,1))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = Conv2D(k,(1,1),strides = (1,1))(x)
    return x

def Dense_Block(x,k):
    
    x1 = Dense_Layer(x,k)
    x1_add = keras.layers.Concatenate()([x1,x])
    x2 = Dense_Layer(x1_add,k)
    x2_add = keras.layers.Concatenate()([x1,x2])
    x3 = Dense_Layer(x2_add,k)
    x3_add = keras.layers.Concatenate()([x1,x2,x3])
    x4 = Dense_Layer(x3_add,k)
    x4_add = keras.layers.Concatenate()([x1,x2,x3,x4])
    x5 = Dense_Layer(x4_add,k)
    x5_add = keras.layers.Concatenate()([x1,x2,x3,x4,x5])
    x6 = Dense_Layer(x5_add,k)
    x6_add = keras.layers.Concatenate()([x,x1,x2,x3,x4,x5,x6])
    return x6_add
    
def Transition_Block(x,k):
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = Conv2D(4*k,(1,1),strides = (1,1))(x)
    x = AveragePooling2D((2,2),strides = (2,2))(x)
    return x


def DenseNet(input_shape=(224,224,3),classes = 6,k=32):
    x_input = Input(input_shape)
    x = ZeroPadding2D((3,3))(x_input)
    x = Conv2D(64,(7,7),strides = 2)(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((1,1),strides = (2,2))(x)
    
    x = Dense_Block(x,k)
    x = Transition_Block(x,k)
    x = Dense_Block(x,k)
    x = Transition_Block(x,k)
    x = Dense_Block(x,k)
    x = Transition_Block(x,k)
    x = Dense_Block(x,k)
    
    x = AveragePooling2D((7,7), strides = (2,2))(x)
    x = Flatten()(x)
    x = Dense(classes, activation = 'softmax')(x)
    model = Model(inputs = x_input,outputs = x)
    return model
    
model = DenseNet()
model.compile(optimizer = "adam",loss = "categorical_crossentropy", metrics = ["accuracy"])
model.summary()
    
    