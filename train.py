import numpy as np
import os
from load_data import load_tag, seq2matrix, batch_generator
from model import generator_model, discriminator_model, gan_model
from keras.optimizers import Adam, RMSprop



IMG_NPY_PATH = 'img.npy'
TAG_PATH = 'tags_clean.csv'
NOISE_DIM = 100
EPOCH = 500
BATCH_SIZE = 64
CKPT_DIR = 'ckpt'
LOG_FILE_PATH = 'history.txt'


def main():
    	
	imgs = np.load(IMG_NPY_PATH)	
	tags = load_tag(TAG_PATH)
	tags_onehot = seq2matrix(tags)
	data_generator = batch_generator(imgs, tags_onehot, noise_dim=NOISE_DIM, batch_size=BATCH_SIZE)

	g_model = generator_model()
	d_model = discriminator_model()
	gan = gan_model(d_model, g_model)

	opt_g = Adam(lr = 0.0002, clipvalue=0.001)
	opt_d = Adam(lr = 0.0002, clipvalue=0.001)

	g_model.compile(loss='mse', optimizer=opt_g)
	d_model.trainable = False
	gan.compile(loss='categorical_crossentropy', optimizer=opt_g, metrics=['acc'])
	d_model.trainable = True
	d_model.compile(loss='categorical_crossentropy', optimizer=opt_d, metrics=['acc'])

	step_per_epoch = int(imgs.shape[0] / BATCH_SIZE)

	d_loss_hist = []
	g_loss_hist = []

	total_itr = 0


	for i in range(EPOCH):

		for j in range(step_per_epoch):

			noise, real_img, right_tag, wrong_tag = next(data_generator)
			fake_img = g_model.predict([noise, right_tag])

			# Train discriminator
			x_img = np.vstack([real_img, fake_img, real_img])
			x_tag = np.vstack([right_tag, right_tag, wrong_tag])
			y_d = np.zeros((3*BATCH_SIZE, 2))
			y_d[:BATCH_SIZE,1] = 1
			y_d[BATCH_SIZE:,0] = 1
			d_loss, d_acc = d_model.train_on_batch([x_img, x_tag], y_d)

			# Train generator
			x_noise = np.random.normal(size=(BATCH_SIZE, NOISE_DIM))
			y_g = np.zeros((BATCH_SIZE,2))
			y_g[:,1] = 1
			d_model.trainable = False
			g_loss, g_acc = gan.train_on_batch([x_noise, right_tag], y_g)
			d_model.trainable = True

			print('ep{}, itr{}, d_loss:{:.4f}, g_loss:{:.4f}, d_acc:{:.4f}, g_acc:{:.4f}'.format(i,j,d_loss,g_loss,d_acc,g_acc))

			d_loss_hist.append(d_loss)
			g_loss_hist.append(g_loss)

			total_itr += 1

			if (total_itr%100)==0:
				g_model.save_weights(os.path.join(CKPT_DIR, 'g_model_ckpt{}.h5'.format(total_itr)))
				d_model.save_weights(os.path.join(CKPT_DIR, 'd_model_ckpt{}.h5'.format(total_itr)))

		
		with open(LOG_FILE_PATH, 'a') as f:
			for ii in range(len(d_loss_hist)):
				f.write('{},{}\n'.format(d_loss_hist[ii], g_loss_hist[ii]))

		d_loss_hist = []
		g_loss_hist = []


if __name__ == "__main__":
	main()