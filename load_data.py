import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize



hair = ['orange hair','white hair','aqua hair','gray hair',
    'green hair','red hair','purple hair','pink hair',
    'blue hair','black hair','brown hair','blonde hair']

eyes = ['gray eyes','black eyes','orange eyes','pink eyes',
    'yellow eyes','aqua eyes','purple eyes','green eyes',
    'brown eyes','red eyes','blue eyes']

tag_dict = {key:i for i,key in enumerate(hair+eyes)}


def load_tag(tagpath):

    global tag_dict
    tags = []

    with open(tagpath, mode='r', encoding='utf-8') as f:
        tag_raw = f.readlines()

    for line in tag_raw:
        valid_tag = []
        tag_str = line.strip('\n').split(',')[1]
        for t in tag_str.split('\t')[:-1]:
            # print(t.split(':')[0].strip((' ')))
            tag = tag_dict.get(t.split(':')[0].strip((' ')), None)
            if tag!=None:
                valid_tag.append(tag_dict[t.split(':')[0].strip(' ')])

        tags.append(valid_tag)
    return tags


def load_img(img_dir):
    
    img_names = os.listdir(img_dir)
    imgs = np.zeros((len(img_names),64,64,3)).astype(np.uint8)
    imgss = np.load("img.npy")
    

    for i, img_name in enumerate(img_names):
        img_arr = resize(imread(os.path.join(img_dir,img_name)), (64,64,3),  preserve_range=True).astype(np.uint8)
        imgs[i] = img_arr
    # np.save("imgs.npy", imgs)

    return imgs
    

def seq2matrix(seqs, class_num=23):

	matrix = np.zeros((len(seqs), class_num))
	for i, sample in enumerate(seqs):
		if len(sample)>0:
			for j in sample:
				matrix[i,j]=1
	return matrix


def batch_generator(img, tag, noise_dim, batch_size):
	
	while True:
		random_idx = np.random.permutation(img.shape[0])
		real_img = img[random_idx[:batch_size]]

		noise = np.random.normal(size=(batch_size, noise_dim))

		right_tag = tag[random_idx[:batch_size]]
		wrong_tag = tag[random_idx[batch_size:2*batch_size]]
		# fake_img = g_model.predict([noise, right_tag])

		yield noise, real_img, right_tag, wrong_tag

