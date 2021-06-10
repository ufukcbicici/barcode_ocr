import numpy as np
import cv2
import tensorflow as tf
import os
import pickle
from scipy.stats import norm
from sklearn import metrics
from sklearn.metrics import classification_report
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from tqdm import tqdm

from utils import Utils


class ConvAE:
    def __init__(self, model_name, model_path,
                 original_dim, latent_dim, conv_layers, dense_layers):
        # Encoder
        self.modelName = model_name
        self.modelPath = model_path
        self.originalDim = original_dim
        self.x = tf.keras.Input(shape=original_dim, name="x_input")
        net = self.x
        for kernel_size, feature_map_count in conv_layers:
            net = tf.keras.layers.Conv2D(feature_map_count, kernel_size, padding="same")(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
            net = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)(net)
        self.lastConvShape = net.shape[1:]
        net = tf.keras.layers.Flatten()(net)
        for hidden_dim in dense_layers:
            net = tf.keras.layers.Dense(hidden_dim, activation=None)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)
        self.output = tf.keras.layers.Dense(2, activation=None)(net)
        self.model = tf.keras.Model(inputs=self.x, outputs=self.output, name=self.modelName)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.mseTracker = tf.keras.metrics.Mean(name="mse_tracker")
        self.imageDict = {}
        self.maskDict = {}
        self.maskObbDict = {}



    # def get_mse(self, x):
    #     x_hat = self.model(x)
    #     delta_x = x_hat - x
    #     squared_delta_x = tf.square(delta_x)
    #     mse = tf.reduce_mean(squared_delta_x)
    #     return mse
    #
    # def get_mse_vector(self, x):
    #     x_hat = self.model(x)
    #     delta_x = x_hat - x
    #     squared_delta_x = tf.square(delta_x)
    #     mse = tf.reduce_mean(squared_delta_x, axis=(1, 2, 3))
    #     return mse
    #
    # def train_step(self, x):
    #     with tf.GradientTape() as tape:
    #         x_hat = self.model(x)
    #         delta_x = x_hat - x
    #         squared_delta_x = tf.square(delta_x)
    #         mse = tf.reduce_mean(squared_delta_x)
    #     grads = tape.gradient(mse, self.model.trainable_variables)
    #     self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #     self.mseTracker.update_state(mse)
    #     print("Mse:{0}".format(self.mseTracker.result().numpy()))
    #
    # def save_model(self, path):
    #     self.model.save(path)
    #
    # def load_model(self, path):
    #     self.model = tf.keras.models.load_model(path)
    #
    # def crop_patch(self, image, sample_coords, verbose=False):
    #     destination_patch = np.zeros(shape=self.originalDim, dtype=image.dtype)
    #     source_left = int(max(int(sample_coords[0]) - self.originalDim[1] / 2, 0))
    #     source_top = int(max(int(sample_coords[1]) - self.originalDim[0] / 2, 0))
    #     source_right = int(min(int(sample_coords[0]) + self.originalDim[1] / 2, image.shape[1]))
    #     source_bottom = int(min(int(sample_coords[1]) + self.originalDim[0] / 2, image.shape[0]))
    #     source_patch = image[source_top:source_bottom, source_left:source_right, :]
    #     destination_left = int(0.5 * (destination_patch.shape[1] - source_patch.shape[1]))
    #     destination_top = int(0.5 * (destination_patch.shape[0] - source_patch.shape[0]))
    #     destination_patch[destination_top:destination_top + source_patch.shape[0],
    #     destination_left:destination_left + source_patch.shape[1]] = source_patch
    #     if verbose:
    #         image_clone = image.copy()
    #         cv2.rectangle(image_clone, (source_left, source_top), (source_right, source_bottom), color=(0, 255, 0))
    #         cv2.imshow("source", image_clone)
    #         cv2.imshow("destination_patch", destination_patch)
    #         cv2.waitKey(0)
    #     return destination_patch
    #
    # def sample_patches_from_image(self, img_path, sample_per_image, verbose=False):
    #     file_path = img_path.numpy().decode("utf-8")
    #     root_path = os.path.split(file_path)[0]
    #     file_name = os.path.split(file_path)[1][:-4]
    #     # Read images, masks and oriented bounding boxes
    #     if file_path not in self.imageDict:
    #         assert file_path not in self.maskDict
    #         assert file_path not in self.maskObbDict
    #         image = cv2.imread(file_path)
    #         mask_image = cv2.imread(file_path[:-4] + "_mask.png")
    #         pickle_path = os.path.join(root_path, "{0}_mask_obb_coordinates.dat".format(file_name))
    #         with open(pickle_path, "rb") as f:
    #             obb_coords = pickle.load(f)
    #         self.imageDict[file_path] = image
    #         self.maskDict[file_path] = mask_image
    #         self.maskObbDict[file_path] = obb_coords
    #     else:
    #         assert file_path in self.maskDict
    #         assert file_path in self.maskObbDict
    #         image = self.imageDict[file_path]
    #         mask_image = self.maskDict[file_path]
    #         obb_coords = self.maskObbDict[file_path]
    #     if verbose:
    #         Utils.show_text_detections(image=image, detections=[obb_coords], window_name="image")
    #         Utils.show_text_detections(image=mask_image, detections=[obb_coords], window_name="mask_image")
    #         cv2.waitKey(0)
    #     # Sample patches from the image, in the pixels of the oriented bounding box
    #     dx = obb_coords[1] - obb_coords[0]
    #     dy = obb_coords[3] - obb_coords[0]
    #     patches = []
    #     for patch_id in range(sample_per_image):
    #         sample_dx = np.random.uniform(low=0.0, high=1.0) * dx
    #         sample_dy = np.random.uniform(low=0.0, high=1.0) * dy
    #         sample_coords = obb_coords[0] + sample_dx + sample_dy
    #         destination_patch = self.crop_patch(image=image, sample_coords=sample_coords, verbose=verbose)
    #         patches.append(destination_patch)
    #     return patches
    #
    # def train(self, train_paths, test_paths, batch_size, patch_per_image, epoch_count):
    #     train_iterator = \
    #         tf.data.Dataset.from_tensor_slices(train_paths).shuffle(buffer_size=100).batch(
    #             batch_size=batch_size)
    #     self.imageDict = {}
    #     self.maskDict = {}
    #     self.maskObbDict = {}
    #
    #     root_path = os.path.join(self.modelPath, self.modelName)
    #     Utils.create_directory(path=root_path)
    #
    #     with tf.device("GPU"):
    #         for epoch_id in range(epoch_count):
    #             self.mseTracker.reset_states()
    #             for iteration_id, batch_paths in enumerate(train_iterator):
    #                 patch_list = []
    #                 for sample_id in range(batch_paths.shape[0]):
    #                     image_patches = self.sample_patches_from_image(img_path=batch_paths[sample_id],
    #                                                                    sample_per_image=patch_per_image, verbose=False)
    #                     patch_list.extend(image_patches)
    #                 X = np.stack(patch_list, axis=0)
    #                 X = (1.0 / 255.0) * X
    #                 with tf.GradientTape() as tape:
    #                     mse = self.get_mse(x=X)
    #                 grads = tape.gradient(mse, self.model.trainable_variables)
    #                 self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    #                 self.mseTracker.update_state(mse)
    #                 print("Epoch:{0} Iteration:{1} Mse:{2}".format(epoch_id, iteration_id,
    #                                                                self.mseTracker.result().numpy()))
    #             self.save_model(path=os.path.join(root_path, "model_epoch{0}".format(epoch_id)))
    #             profiling_file = open(os.path.join(root_path, 'fit_ae.txt'), 'a')
    #             profiling_file.write("Epoch:{0} Mse:{1}\n".format(epoch_id, self.mseTracker.result().numpy()))
    #             profiling_file.close()
    #
    # def train_anomaly_detector(self, train_paths, batch_size):
    #     with tf.device("GPU"):
    #         mse_list = []
    #         patch_list = []
    #         for file_path in tqdm(train_paths):
    #             image = cv2.imread(file_path)
    #             root_path = os.path.split(file_path)[0]
    #             file_name = os.path.split(file_path)[1][:-4]
    #             image_mse_list = []
    #             pickle_path = os.path.join(root_path, "{0}_mask_obb_coordinates.dat".format(file_name))
    #             with open(pickle_path, "rb") as f:
    #                 obb_coords = pickle.load(f)
    #             polygon = Polygon([obb_coords[0], obb_coords[1], obb_coords[2], obb_coords[3]])
    #             for i in range(image.shape[0]):
    #                 for j in range(image.shape[1]):
    #                     point = Point(j, i)
    #                     does_contain = polygon.contains(point)
    #                     if does_contain:
    #                         destination_patch = self.crop_patch(image=image,
    #                                                             sample_coords=np.array([j, i]),
    #                                                             verbose=False)
    #                         destination_patch = (1.0 / 255.0) * destination_patch
    #                         patch_list.append(destination_patch)
    #                     if len(patch_list) == batch_size:
    #                         X = np.stack(patch_list, axis=0)
    #                         mse_vector = self.get_mse_vector(x=X).numpy()
    #                         image_mse_list.append(mse_vector)
    #                         patch_list = []
    #             if len(patch_list) > 0:
    #                 X = np.stack(patch_list, axis=0)
    #                 mse_vector = self.get_mse_vector(x=X).numpy()
    #                 image_mse_list.append(mse_vector)
    #                 patch_list = []
    #             image_mse_list = np.concatenate(image_mse_list)
    #             mse_list.append(image_mse_list)
    #     all_mses = np.concatenate(mse_list)
    #     print("Mean MSE:{0}".format(np.mean(all_mses)))
    #     mse_values_path = os.path.join(self.modelPath, self.modelName, 'mse_values.dat')
    #     with open(mse_values_path, "wb") as f:
    #         pickle.dump(all_mses, f)
    #
    # def eval_anomaly_detector(self, test_paths, batch_size, percentile):
    #     mse_values_path = os.path.join(self.modelPath, self.modelName, 'mse_values.dat')
    #     with open(mse_values_path, "rb") as f:
    #         all_mses = pickle.load(f)
    #     mses_sorted = np.sort(all_mses)
    #     print("X")
