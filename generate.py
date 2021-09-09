import argparse
import numpy as np
from keras.models import load_model
from load_data import tag_dict
from skimage.io import imsave



hair = ['orange hair','white hair','aqua hair','gray hair',
    'green hair','red hair','purple hair','pink hair',
    'blue hair','black hair','brown hair','blonde hair']

eye = ['gray eyes','black eyes','orange eyes','pink eyes',
    'yellow eyes','aqua eyes','purple eyes','green eyes',
    'brown eyes','red eyes','blue eyes']



tag_dict = {key:i for i,key in enumerate(hair+eye)}




def eye_color_checker(color):
    if color and color+' eyes' not in eye:
        raise argparse.ArgumentTypeError('invalid eye color')
    return color

def hair_color_checker(color):
    if color and color+' hair' not in hair:
        raise argparse.ArgumentTypeError('invalid hair color')
    return color




if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="increase output verbosity",
                        action="store_true")
    parser = argparse.ArgumentParser()
    parser.add_argument("--hair", help="hair color", type=hair_color_checker)
    parser.add_argument("--eye", help="eye color", type=eye_color_checker)
    parser.add_argument("output_path")

    args = parser.parse_args()

    v = np.random.normal(size=(1, 100))
    condition = np.zeros((1, 23))
    if args.hair:
        condition[0, tag_dict[args.hair+" hair"]] = 1
    if args.eye:
        condition[0, tag_dict[args.eye+" eyes"]] = 1


    model = load_model('gan_model.h5')
    img = model.predict([v, condition])[0]
    imsave(args.output_path, img)