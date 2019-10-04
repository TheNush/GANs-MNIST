import numpy as np
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, Dense, Flatten, Reshape, Conv2DTranspose
from keras.optimizers import Adam

def the_discriminator(input_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=input_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	optimizer = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

# model = the_discriminator()
# model.summary()

def real_digits():
	(x_train, _), (_, _) = load_data()
	train = np.expand_dims(x_train, axis=-1)
	train = train.astype('float32')
	train /= 255.0
	return train

def gen_real_digits(df, n_digits):
	index = np.random.randint(0, df.shape[0], n_digits)
	x = df[index]
	y = np.ones((n_digits, 1))
	return x, y

def gen_fake_digits(g_model, latent_space_dim, n_digits):
	# x = np.random.randn(28 * 28 * n_digits)
	inputs = gen_latent_points(latent_space_dim, n_digits)
	# x = x.reshape(n_digits, 28, 28, 1)
	x = g_model.predict(inputs)
	y = np.zeros((n_digits, 1))
	return x, y

def train_the_discriminator(model, df, n_iterations=40, batch_size=256):
	half_batch = int(batch_size/2)
	for i in range(n_iterations):
		x_real, y_real = gen_real_digits(df, half_batch)
		_, real_acc = model.train_on_batch(x_real, y_real)
		x_fake, y_fake = gen_fake_digits(half_batch)
		_, fake_acc = model.train_on_batch(x_fake, y_fake)
		print('Iteration: {}	real_accuracy={}	fake_accuracy={}'.format(i+1, real_acc, fake_acc))

def the_generator(latent_space_dim):
	model = Sequential()
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_space_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7,7,128)))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
	return model

def gen_latent_points(latent_space_dim, n_digits):
	inputs = np.random.randn(latent_space_dim * n_digits)
	inputs = inputs.reshape(n_digits, latent_space_dim)
	return inputs

# latent_space_dim = 100
# model = the_generator(latent_space_dim)
# n_digits = 15
# x, _ = gen_fake_digits(model, latent_space_dim, n_digits)

# for i in range(n_digits):
# 	plt.subplot(5,3, i+1)
# 	plt.axis('off')
# 	plt.imshow(x[i,:,:,0], cmap='gray_r')
# plt.show()

def gan_model(g_model, d_model):
	d_model.trainable = False
	model = Sequential()
	model.add(g_model)
	model.add(d_model)
	optimizer = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	return model

def train_gan(gan_model, latent_space_dim, n_epochs=100, batch_size=256):
	for i in range(n_epochs):
		inputs_gan = gen_latent_points(latent_space_dim, batch_size)
		y_gan = np.ones((batch_size, 1))
		gan_model.train_on_batch(inputs_gan, y_gan)

def save_orig(examples, n=10):
	for i in range(n * n):
		plt.subplot(n, n, i+1)
		plt.axis('off')
		plt.imshow(examples[i,:,:,0], cmap='gray_r')
	filename = 'original_images.png'
	plt.savefig(filename)
	plt.close()

def save_plot(examples, epoch, n=10):
	for i in range(n * n):
		plt.subplot(n, n, i+1)
		plt.axis('off')
		plt.imshow(examples[i,: ,:, 0], cmap='gray_r')
	filename = 'generated_plot_e{}.png'.format(epoch + 1)
	plt.savefig(filename)
	plt.close()

def train_prime(g_model, d_model, gan_model, df, latent_space_dim, n_epochs=100, batch_size=256):
	batch_per_epoch = int(df.shape[0]/batch_size)
	half_batch = int(batch_size/2)
	for i in range(n_epochs):
		for j in range(batch_per_epoch):
			x_real, y_real = gen_real_digits(df, half_batch)
			x_fake, y_fake = gen_fake_digits(g_model, latent_space_dim, half_batch)
			x, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
			d_loss, _ = d_model.train_on_batch(x, y)
			inputs_gan = gen_latent_points(latent_space_dim, batch_size)
			y_gan = np.ones((batch_size, 1))
			g_loss = gan_model.train_on_batch(inputs_gan, y_gan)
			print("Epoch {}:	{}/{}batches 	disc_loss: {}	gen_loss: {}".format(i+1, j+1, batch_per_epoch, d_loss, g_loss))
			if (i+1) % 10 == 0:
				tell_me_the_story(i, g_model, d_model, df, latent_space_dim)

def tell_me_the_story(epoch, g_model, d_model, df, latent_space_dim, n_digits=100):
	x_real, y_real = gen_real_digits(df, n_digits)
	_, real_acc = d_model.evaluate(x_real, y_real, verbose=0)
	x_fake, y_fake = gen_fake_digits(g_model, latent_space_dim, n_digits)
	_, fake_acc = d_model.evaluate(x_fake, y_fake, verbose=0)
	print("Accuracies:\n Real: {}	Fake: {}".format(real_acc, fake_acc))
	save_plot(x_fake, epoch)
	filename = 'generator_model_{}.h5'.format(epoch + 1)
	g_model.save(filename)

latent_space_dim = 100
d_model = the_discriminator()
g_model = the_generator(latent_space_dim)
gan_model = gan_model(g_model, d_model)
df = real_digits()
# save_orig(gen_real_digits(df, 128)[0])
train_prime(g_model, d_model, gan_model, df, latent_space_dim)
# model = the_discriminator()
# df = real_digits()
# train_the_discriminator(model, df)
