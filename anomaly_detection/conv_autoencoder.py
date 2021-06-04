import numpy as np
import cv2
import tensorflow as tf
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import classification_report


class ConvAE:
    def __init__(self, model_name, model_path,
                 original_dim, latent_dim, layers_encoder, layers_decoder):
        # Encoder
        self.modelName = model_name
        self.modelPath = model_path
        self.originalDim = original_dim
        self.x = tf.keras.Input(shape=original_dim, name="x_input")
        net = self.x
        for kernel_size, feature_map_count in layers_encoder:
            net = tf.keras.layers.Conv2D(feature_map_count, kernel_size, padding="same")(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
            net = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(net)
        self.lastConvShape = net.shape[1:]
        net = tf.keras.layers.Flatten()(net)
        self.flatDim = net.shape[-1]
        net = tf.keras.layers.Dense(latent_dim, name="latent")(net)
        self.z = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
        # Decoder
        net = self.z
        net = tf.keras.layers.Dense(self.flatDim, name="latent_back")(net)
        net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
        net = tf.keras.layers.Reshape(target_shape=self.lastConvShape)(net)
        for kernel_size, feature_map_count in layers_decoder:
            net = tf.keras.layers.Conv2DTranspose(feature_map_count, kernel_size, padding="same", strides=2)(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
        net = tf.keras.layers.Conv2DTranspose(3, 3, strides=1, padding="same")(net)
        self.xHat = net
        self.model = tf.keras.Model(inputs=self.x, outputs=self.xHat, name=self.modelName)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.mseTracker = tf.keras.metrics.Mean(name="mse_tracker")
        self.imageDict = {}
        self.maskDict = {}

    def get_mse(self, x):
        x = tf.expand_dims(tf.cast(x, tf.float32), axis=-1)
        x_hat = self.model(x)
        delta_x = x_hat - x
        squared_delta_x = tf.square(delta_x)
        mse_vector = tf.reduce_mean(squared_delta_x, axis=(1, 2))
        return mse_vector

    def train_step(self, x):
        x = tf.expand_dims(x, axis=-1)
        with tf.GradientTape() as tape:
            x_hat = self.model(x)
            delta_x = x_hat - x
            squared_delta_x = tf.square(delta_x)
            mse = tf.reduce_mean(squared_delta_x)
        grads = tape.gradient(mse, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.mseTracker.update_state(mse)
        print("Mse:{0}".format(self.mseTracker.result().numpy()))

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def sample_patches_from_image(self, img_path, sample_per_image):
        file_path = img_path.numpy().decode("utf-8")
        if file_path not in self.imageDict:
            assert file_path not in self.maskDict
            image = cv2.imread(file_path)
            mask_image = cv2.imread(file_path[:-4] + "_mask.png")
            self.imageDict[file_path] = image
            self.maskDict[file_path] = mask_image
        else:
            assert file_path in self.maskDict
            image = self.imageDict[file_path]
            mask_image = self.maskDict[file_path]
        # Get non zero positions
        non_zero_indices = np.nonzero(mask_image)
        print("X")

    def train(self, bounding_box_dict, train_paths, test_paths, batch_size, patch_per_image, epoch_count):
        train_iterator = \
            tf.data.Dataset.from_tensor_slices(train_paths).shuffle(buffer_size=100).batch(
                batch_size=batch_size).map(lambda x: self.sample_patches_from_image(x, patch_per_image))
        self.imageDict = {}
        self.maskDict = {}

        with tf.device("GPU"):
            for iteration_id, batch_paths in range(train_iterator):
                print("X")


# if __name__ == "__main__":
#
#     # *************************** DATA 1 ***************************
#     # epoch_count = 100
#     # # data_dim = 50
#     # hidden_dim = 16
#     # batch_size = 128
#     # encoder_layers = [(5, 8), (5, 16)]
#     # decoder_layers = [(5, 16), (5, 8)]
#     # np.random.seed(42)
#     # data_generator = Data1Reader(batch_size=batch_size, path='../anomaly-data-1')
#     # conv_ae = ConvAE(name="conv_ae_data1",
#     #                  original_dim=data_generator.trainX.shape[1],
#     #                  latent_dim=hidden_dim,
#     #                  layers_encoder=encoder_layers,
#     #                  layers_decoder=decoder_layers)
#     #
#     # # for epoch_id in range(epoch_count):
#     # #     for batch_X, batch_y in data_generator.trainSet:
#     # #         print("Epoch:{0}".format(epoch_id))
#     # #         conv_ae.train_step(x=tf.cast(batch_X, tf.float32))
#     # # conv_ae.save_model()
#     # conv_ae.load_model()
#     #
#     # mean_normal = None
#     # std_normal = None
#     # mean_anormal = None
#     # std_anormal = None
#     # for data_type, x_, y_ in (["train", data_generator.trainX, data_generator.trainy],
#     #                           ["test", data_generator.testX, data_generator.testy]):
#     #
#     #     if data_type == "train":
#     #         loss_vec = conv_ae.get_mse(x_)
#     #         losses_normal = loss_vec[y_ == 0]
#     #         losses_anormal = loss_vec[y_ == 1]
#     #         mean_normal = np.mean(losses_normal)
#     #         std_normal = np.std(losses_normal)
#     #         mean_anormal = np.mean(losses_anormal)
#     #         std_anormal = np.std(losses_anormal)
#     #
#     #     loss_vec = conv_ae.get_mse(x_)
#     #     norm_p = norm.pdf(loss_vec, loc=mean_normal, scale=std_normal)
#     #     anorm_p = norm.pdf(loss_vec, loc=mean_anormal, scale=std_anormal)
#     #     predictions = np.stack([norm_p, anorm_p], axis=1)
#     #     y_hat = np.argmax(predictions, axis=1)
#     #     print("****************** Data:{0} ******************".format(data_type))
#     #     print(classification_report(y_, y_hat))
#     #     fpr, tpr, thresholds = metrics.roc_curve(y_, y_hat, pos_label=1)
#     #     auc = metrics.auc(fpr, tpr)
#     #     print("auc={0}".format(auc))
#     # print("X")
#     # *************************** DATA 1 ***************************
#
#     # *************************** DATA 2 ***************************
#     epoch_count = 100
#     # data_dim = 50
#     hidden_dim = 4
#     batch_size = 128
#     encoder_layers = [(3, 8)]
#     decoder_layers = [(3, 8)]
#     np.random.seed(42)
#     data_generator = Data2Reader(batch_size=batch_size, path='../anomaly-data-2.mat')
#     conv_ae = ConvAE(name="conv_ae_data2",
#                      original_dim=data_generator.trainX.shape[1],
#                      latent_dim=hidden_dim,
#                      layers_encoder=encoder_layers,
#                      layers_decoder=decoder_layers)
#
#     for epoch_id in range(epoch_count):
#         for batch_X, batch_y in data_generator.trainSet:
#             print("Epoch:{0}".format(epoch_id))
#             conv_ae.train_step(x=tf.cast(batch_X, tf.float32))
#     conv_ae.save_model()
#     # conv_ae.load_model()
#
#     mean_normal = None
#     std_normal = None
#     mean_anormal = None
#     std_anormal = None
#     for data_type, x_, y_ in (["train", data_generator.trainX, data_generator.trainy],
#                               ["test", data_generator.testX, data_generator.testy]):
#
#         if data_type == "train":
#             loss_vec = conv_ae.get_mse(x_)
#             losses_normal = loss_vec[y_ == 0]
#             losses_anormal = loss_vec[y_ == 1]
#             mean_normal = np.mean(losses_normal)
#             std_normal = np.std(losses_normal)
#             mean_anormal = np.mean(losses_anormal)
#             std_anormal = np.std(losses_anormal)
#
#         loss_vec = conv_ae.get_mse(x_)
#         norm_p = norm.pdf(loss_vec, loc=mean_normal, scale=std_normal)
#         anorm_p = norm.pdf(loss_vec, loc=mean_anormal, scale=std_anormal)
#         predictions = np.stack([norm_p, anorm_p], axis=1)
#         y_hat = np.argmax(predictions, axis=1)
#         print("****************** Data:{0} ******************".format(data_type))
#         print(classification_report(y_, y_hat))
#         fpr, tpr, thresholds = metrics.roc_curve(y_, y_hat, pos_label=1)
#         auc = metrics.auc(fpr, tpr)
#         print("auc={0}".format(auc))
#     print("X")
#     # *************************** DATA 2 ***************************