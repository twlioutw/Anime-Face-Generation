import numpy as np
import os
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dense, Dropout,Flatten, Input, LeakyReLU, RepeatVector, Reshape
from keras.optimizers import Adam, RMSprop
import keras.backend as K



def generator_model(noise_dim=100, condition_dim=23):
	kernel = (3,3)
	noise_input = Input(shape=(noise_dim,))
	condition_input = Input(shape=(condition_dim,))
	
	x = Concatenate(axis=-1)([noise_input, condition_input])
	x = Dense(units=32*32*128)(x)
	x = BatchNormalization(axis=-1)(x)
	x = LeakyReLU(0.2)(x)
	x = Dropout(0.5)(x)

	x = Reshape((32,32,128))(x)

	x = Conv2DTranspose(128, kernel, padding='same',strides=(2,2))(x)
	x = BatchNormalization(axis=-1)(x)
	x = LeakyReLU(0.2)(x)
	x = Dropout(0.3)(x)

	x = Conv2D(64, kernel, padding='same')(x)
	x = BatchNormalization(axis=-1)(x)
	x = LeakyReLU(0.2)(x)
	x = Dropout(0.3)(x)

	x = Conv2D(32, kernel, padding='same')(x)
	x = BatchNormalization(axis=-1)(x)
	x = LeakyReLU(0.2)(x)
	x = Dropout(0.3)(x)

	x = Conv2D(3, kernel, padding='same', activation='sigmoid')(x)

	g_model = Model([noise_input, condition_input], x)

	return g_model



def discriminator_model(condition_dim=23):

	img_input = Input(shape=(64,64,3))
	kernel = (5,5)

	x = img_input
	for i in range(1,4):
		x = Conv2D((2**i)*64, kernel, padding='same', strides=(2,2))(x)
		x = LeakyReLU(0.2)(x)
		x = Dropout(0.3)(x)


	condition_input = Input(shape=(condition_dim,))

	x = Flatten()(x)
	x = Concatenate(axis=-1)([x, condition_input])

	x = Dense(units=256)(x)
	x = LeakyReLU(0.2)(x)
	x = Dense(units=2, activation='softmax')(x)

	d_model = Model([img_input, condition_input], x)

	return d_model


def gan_model(d_model, g_model, noise_dim=100, condition_dim=23):
	noise_input = Input(shape=(noise_dim,))

	condition_input = Input(shape=(condition_dim,))
	generated_image = g_model([noise_input, condition_input])
	gan_output = d_model([generated_image, condition_input])
	gan = Model(inputs=[noise_input, condition_input],outputs=[gan_output])

	return gan
