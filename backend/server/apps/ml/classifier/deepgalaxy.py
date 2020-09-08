import joblib
import pandas as pd
import glob
import cv2
import os 
import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from cvae import VariationalAutoEncoder
from django.conf import settings

class DeepGalaxyClassifier:
    def __init__(self):
        # path_to_artifacts = "../../research/"
        # self.values_fill_missing =  joblib.load(path_to_artifacts + "train_mode.joblib")
        # self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")
        # self.model = joblib.load(path_to_artifacts + "extra_trees.joblib")
        vae = VariationalAutoEncoder(input_shape=(512,512,1), latent_dim=32)
        vae.create_encoder()
        vae.create_decoder()
        bash_dir = os.path.dirname(__file__)
        vae.load_weights(os.path.join(bash_dir, "vae.h5"))
        vae.build(input_shape=(None,512,512,1))
        self.model = vae

        with h5py.File(os.path.join(bash_dir, 'encoded_train_images.h5'), 'r') as h5f:  # compressed training dataset
            self.enc_train = h5f['encoded'][()]


    def preprocessing(self, image_path):
        # # JSON to pandas DataFrame
        # input_data = pd.DataFrame(input_data, index=[0])
        # # fill missing values
        # input_data.fillna(self.values_fill_missing)
        # # convert categoricals
        # for column in [
        #     "workclass",
        #     "education",
        #     "marital-status",
        #     "occupation",
        #     "relationship",
        #     "race",
        #     "sex",
        #     "native-country",
        # ]:
        #     categorical_convert = self.encoders[column]
        #     input_data[column] = categorical_convert.transform(input_data[column])
        # bash_dir = os.path.dirname(__file__)
        img_list = glob.glob(image_path)
        print(img_list)
        img_size = 512

        obs_imgs = []
        obs_imgs_title = []
        for img_name in img_list:
            im = cv2.imread(img_name)
            im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_resized = cv2.resize(im_grey, (img_size,img_size))
            print(img_name, im_resized.shape)
            obs_imgs.append((im_resized.reshape(img_size,img_size,1)/255).astype(np.float32))
            obs_imgs_title.append(os.path.basename(img_name))
        print(obs_imgs_title)
        obs_imgs = np.array(obs_imgs)

        return obs_imgs

    def predict(self, model, query_img_array, query_img_id, k=5):
        enc_test = model.encoder.predict(query_img_array)
        print(np.array(enc_test).shape)
        enc_dist = tf.norm(enc_test[query_img_id] - self.enc_train, axis=-1)
        enc_dist_unique, enc_dist_unique_id = np.unique(enc_dist, return_index=True)
        print('enc_train', self.enc_train.shape, self.enc_train[enc_dist_unique_id[0]])
        decoded_image = model.decoder.predict(self.enc_train[enc_dist_unique_id[0]].reshape(1,32))
        print(decoded_image.shape)
    #     print(enc_dist_unique_id)
    #     print(enc_dist_unique)
        panel_size = 12
        
    #     neighbors = tf.argsort(enc_dist)
    #     print(neighbors)
    #     print(tf.sort(enc_dist)[k:])
        # plt.figure(11)
        orig_img = query_img_array[query_img_id]
        decoded_img = decoded_image[0]
        mae_dist = tf.keras.losses.MAE(orig_img, decoded_img).numpy()[0]
        orig_img_tf = tf.image.convert_image_dtype(orig_img, tf.float32)
        decoded_img_tf = tf.image.convert_image_dtype(decoded_img, tf.float32)
        ssim_dist = tf.image.ssim(orig_img_tf, decoded_img_tf, max_val=1.0)
        
        return decoded_image[0, :, :, 0]

        # TODO: seperate image plot function
        # ax = plt.subplot(1, 2, 1)
        # ax.imshow(query_img_array[query_img_id][:, :, 0])
        # ax.set_title('obs')
        # ax = plt.subplot(1, 2, 2)
        # ax.imshow(decoded_image[0, :, :, 0])
        # ax.set_title('rec, %.2f' % ssim_dist)
        # plt.show()
        # plt.tight_layout()
        
        # calculate JS distance
    #     print('dist', tf.keras.losses.MAE(query_img_array[query_img_id][:, :, 0].flatten(), decoded_image[0, :, :, 0].flatten()))

    #     plt.figure(12, figsize=(panel_size, (k+1)*panel_size))

    #     for i in range(1, k+1):
    #         ax = plt.subplot(1, k, i)
    #         ax.imshow(train_images[enc_dist_unique_id[i]][:, :, 0])
    #         ax.set_title('%d, %.2f' % (enc_dist_unique_id[i], enc_dist_unique[i]))
        

    def postprocessing(self, input_data):
        # label = "<=50K"
        # if input_data[1] > 0.5:
        #     label = ">50K"
        # return {"probability": input_data[1], "label": label, "status": "OK"}
        return None

    def compute_prediction(self, image_path):
        # try:
        #     input_data = self.preprocessing(input_data)
        #     prediction = self.predict(input_data)[0]  # only one sample
        #     prediction = self.postprocessing(prediction)
        # except Exception as e:
        #     return {"status": "Error", "message": str(e)}

        # return prediction


        try:
            input_data = self.preprocessing(image_path)
            prediction = self.predict(self.model, input_data, 0, k=6)
            # output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "media/prediction", "prediction.png")
            output_path = settings.MEDIA_ROOT
            print("----output path-------", output_path)
            plt.imsave(os.path.join(output_path, "prediction/prediction.png"), prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        
        return "OK" # TODO: get the probability
