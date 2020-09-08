import tensorflow as tf
import numpy as np
import h5py
from .data_io import DataIO


class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, input_shape=(512,512,1), latent_dim=32, name="autoencoder", **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self._input_shape = input_shape
        self.latent_dim = latent_dim

        # layers
        self.input_layer = tf.keras.layers.Input(shape=input_shape)
        self.base_model = None
        self.encoder = None
        self.decoder = None
        self.n_conv_blocks = 5
        self._z_mean = None
        self._z_log_var = None
        self._z = None

    @staticmethod
    def sample_from_latent_distribution(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z

    def create_encoder(self, base_model=None):
        if base_model is None:
            # create a basic baseline model if `base_model` is not given
            n_filters_head = 16
            intermediate_results = dict()
            for tt in range(self.n_conv_blocks):
                if tt == 0:
                    intermediate_results['%d_Conv2D' % tt] = tf.keras.layers.Conv2D(input_shape=self._input_shape, filters=n_filters_head*(tt+1), kernel_size=3, padding='same', activation='relu')(self.input_layer)
                else:
                    intermediate_results['%d_Conv2D' % tt] = tf.keras.layers.Conv2D(filters=n_filters_head*(tt+1), kernel_size=3, padding='same', activation='relu')(intermediate_results['%d_MaxPooling2D' % (tt-1)])

                intermediate_results['%d_MaxPooling2D' % tt] = tf.keras.layers.MaxPooling2D((2,2))(intermediate_results['%d_Conv2D' % tt])

            base_model = tf.keras.models.Model(self.input_layer, intermediate_results['%d_MaxPooling2D' % (self.n_conv_blocks - 1)], name='base_model')
        else:
            # disable the Batch Normalization layers (if any)

            for layer in base_model.layers:
#                 print(type(layer), isinstance(layer, tf.keras.layers.BatchNormalization))
#                 if isinstance(layer, tf.keras.layers.BatchNormalization):
                if 'BatchNormalization' in str(layer.__class__).split('.')[-1]:
                    # print('disable BN')
                    # layer.trainable = False
                    pass

        self.base_model = base_model

        # determine the number of image channels required by the `base_model`
        n_channels = base_model.input_shape[-1]

        # make sure that the shape of the input model is compatable with the input data
        if self._input_shape[-1] < n_channels:
            repeat_channel = tf.keras.layers.Lambda(lambda x: tf.keras.backend.repeat_elements(x, n_channels, axis=-1), input_shape=self._input_shape)(self.input_layer)
            # obtain the convolutional feature maps from the `base_model`
            conv_base_model = base_model(repeat_channel)
        else:
            conv_base_model = base_model(self.input_layer)

        # flatten the convolutional feature maps
        # conv_flattened = tf.keras.layers.GlobalAveragePooling2D(name='gap')(conv_base_model)
        # conv_flattened = tf.keras.layers.Flatten(name='flatten')(conv_flattened)
        conv_flattened = tf.keras.layers.Flatten(name='flatten')(conv_base_model)

        dropout = tf.keras.layers.Dropout(0.2)(conv_flattened)

        # compress and encode
        z_mean = tf.keras.layers.Dense(self.latent_dim, name='z_mean')(dropout)
        z_log_var = tf.keras.layers.Dense(self.latent_dim, name='z_log_var')(conv_flattened)
        z = tf.keras.layers.Lambda(self.sample_from_latent_distribution, name='z')([z_mean, z_log_var])

        self._z_mean = z_mean
        self._z_log_var = z_log_var

        # create the model
        self.encoder = tf.keras.models.Model(self.input_layer, z, name='encoder')

        return self.encoder

    def create_decoder(self):
        z_inputs = tf.keras.layers.Input(shape=(self.latent_dim,))
        # with the compressed representation `z`, upscale the dimentionality to the uncompressed status
        orig_dim = self.base_model.outputs[0].shape
        expanded = tf.keras.layers.Dense(input_shape=(self.latent_dim,), units=tf.reduce_prod(orig_dim[1:]), name='expand', activation='relu')(z_inputs)
        dropout = tf.keras.layers.Dropout(0.2)(expanded)
        recovered_shape = tf.keras.layers.Reshape(target_shape=orig_dim[1:], name='recovered_shape')(dropout)
#         bn = tf.keras.layers.BatchNormalization()(recovered_shape)

        # figure out how many time should the decoder upsample

        self.n_conv_blocks = int(np.log2(self._input_shape[1] // orig_dim[1]))
        print(self._input_shape[1], orig_dim[1], self.n_conv_blocks)
#         self.n_conv_blocks = 3

        intermediate_results = dict()

        for tt in range(self.n_conv_blocks):
            if tt == 0:
                intermediate_results['%d_Conv2DTranspose' % tt] = tf.keras.layers.Conv2DTranspose(input_shape=orig_dim[1:], filters=64, kernel_size=3, padding='same', activation='relu')(recovered_shape)
            else:
                intermediate_results['%d_Conv2DTranspose' % tt] = tf.keras.layers.Conv2DTranspose(input_shape=orig_dim[1:], filters=64, kernel_size=3, padding='same', activation='relu')(intermediate_results['%d_UpSampling2D' % (tt-1)])

            intermediate_results['%d_UpSampling2D' % tt] = tf.keras.layers.UpSampling2D((2,2))(intermediate_results['%d_Conv2DTranspose' % tt])

        decoded = intermediate_results['%d_UpSampling2D' % (self.n_conv_blocks - 1)]
#         decoded = recovered_shape
        reconstructed = tf.keras.layers.Conv2DTranspose(filters=self.base_model.input_shape[-1], kernel_size=3, padding='same')(decoded)
#         print('dkfskdj', self._input_shape[-1], reconstructed.shape[-1])
#         if reconstructed.shape[-1] > self._input_shape[-1]:
#             # discard the additional repeated layers
#             print('redu')
#             self.decoder = tf.keras.models.Model(z_inputs, reconstructed[:, :, :, :self._input_shape[-1]], name='decoder')
#         else:
#             self.decoder = tf.keras.models.Model(z_inputs, reconstructed, name='decoder')
        self.decoder = tf.keras.models.Model(z_inputs, reconstructed, name='decoder')

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def loss(self, x, x_logit):
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(self._z, 0., 0.)
        logqz_x = self.log_normal_pdf(self._z, self._z_mean, self._z_log_var)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def call(self, inputs):
        z = self.encoder(inputs)
        self._z = z
        reconstructed = self.decoder(z)
        return reconstructed

    def load_model(self, encoder_fn, decoder_fn):
        self.encoder = tf.keras.models.load_model(encoder_fn)
        self.decoder = tf.keras.models.load_model(decoder_fn)

    def save_model(self, encoder_fn, decoder_fn):
        self.encoder.save(encoder_fn)
        self.decoder.save(decoder_fn)

    # def load_weights(self, encoder_fn, decoder_fn):
    #     self.encoder = tf.keras.models.load_weights(encoder_fn)
    #     self.decoder = tf.keras.models.load_weights(decoder_fn)

    # def save_weights(self, encoder_fn, decoder_fn):
    #     self.encoder.save_weights(encoder_fn)
    #     self.decoder.save_weights(decoder_fn)


def ssim_loss(y_true, y_pred):
    # return (1.0 - tf.image.ssim(y_true, y_pred, 1.0)) + tf.keras.losses.MAE(y_true, y_pred)
    mae_pixel = tf.reduce_mean(tf.math.abs(y_true - y_pred), axis=[1,2,3])
    return (1.0 - tf.image.ssim(y_true, y_pred, 1.0)) + mae_pixel * 10
    # return mae_pixel
    # return tf.keras.losses.KLDivergence(y_true, y_pred)

if __name__ == "__main__":
    vae = VariationalAutoEncoder(input_shape=(512,512,1), latent_dim=32)
    # base_model = efn.EfficientNetB0(input_shape=(512,512,1), include_top=False, weights=None)
    # base_model = tf.keras.applications.ResNet50(input_shape=(512,512,1), include_top=False, weights=None)
    # print(base_model.summary())
    # vae.create_encoder(base_model)
    vae.create_encoder()
    vae.create_decoder()
    vae.build(input_shape=(None,512,512,1))
    # vae.base_model.summary()
    vae.encoder.summary()
    vae.decoder.summary()
    vae.summary()


    data_io = DataIO()
    X, Y = data_io.load_partial('../output_bw_512.hdf5', dset_name_pattern='s_*', camera_pos='*')
    # X, Y = data_io.load_partial('../output_bw_512.hdf5', dset_name_pattern='s_1_m_1*', camera_pos=[1,2,3])
    train_images = X.astype(np.float32)

    # vae.compile(optimizer='adam', loss='mae')
    vae.compile(optimizer='adam', loss=ssim_loss)

    vae.fit(train_images, train_images, epochs=3, batch_size=16, shuffle=True)
    # vae.save_weights('cvae_encoder.h5', 'cvae_decoder.h5')
    vae.save_weights('vae.h5')

    encoded_train_images = vae.encoder.predict(train_images, verbose=1)
    with h5py.File('encoded_train_images_efn.h5', 'w') as h5f:
        h5f.create_dataset('encoded', data=encoded_train_images)
