#! -*- coding: utf-8 -*-
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Lambda
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

#训练数据和测试数据获取
def load_mnist():
  #使用keras.datasets载入mnist数据集
  (train_data, _), (test_data, _) = tf.keras.datasets.mnist.load_data()
  #调整train_data的数据维度
  train_data = train_data.reshape((-1, 28 * 28)) / 255.0
  #调整test_data的数据维度
  test_data = test_data.reshape((-1, 28 * 28)) / 255.0

  return train_data,test_data

# 模型搭建
def build_model(params):
    # 编码器
    inputs = Input(shape=params.input_shape, name='encoder_input')
    x = Dense(params.intermediate_dim, activation='relu')(inputs)

    # 算p(Z|X)的均值和方差
    z_mean = Dense(params.latent_dim, name='z_mean')(x)
    z_log_var = Dense(params.latent_dim, name='z_log_var')(x)

    # 重参数技巧
    def sampling(args):
        z_mean, z_log_var = args
        batch_size, latent_dim = K.shape(z_mean)[0], K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch_size, latent_dim))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # 通过重参数方法随机采样样本z
    z = Lambda(sampling, output_shape=(params.latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, z, name='encoder')

    # 解码器
    latent_inputs = Input(shape=(params.latent_dim,), name='z_sampling')
    x = Dense(params.intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(params.input_shape, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # VAE
    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs, name='vae_mlp')

    # 重构loss计算
    if params.mse:
        from keras.losses import mse
        reconstruction_loss = mse(inputs, outputs)
    else:
        from keras.losses import binary_crossentropy
        reconstruction_loss = binary_crossentropy(inputs, outputs)
    # 重构正确性度量
    xent_loss = params.input_shape * reconstruction_loss
    # KL loss计算
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    vae.add_loss(vae_loss)

    # 定义模型训练优化器
    vae.compile(optimizer='rmsprop')
    return vae


# 绘制原始图像和重构图像
def draw(test_data, decoded_imgs, n, pic_name):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # 绘制原始图像
        ax = plt.subplot(2, n, i + 1)

        plt.imshow(test_data[i].reshape(28, 28), cmap='Greys')
        plt.title("Image {}".format(i + 1))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 绘制重构图像
        ax = plt.subplot(2, n, i + 1 + n)

        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='Greys')
        plt.title("Image {}".format(i + 1 + n))

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # 绘制到图片文件'vae.png'
    plt.savefig('{}'.format(pic_name))


def main(params):
    # 数据获取
    x_train, x_test = load_mnist()

    # 模型搭建
    vae = build_model(params)
    # 模型训练
    vae.fit(x_train,
            batch_size=params.batch_size,
            epochs=params.epochs,
            validation_data=(x_test, None),
            verbose=2)
    # 重构图像
    decoded = vae.predict(x_test)
    # 定义展示的原始图像个数
    n = 5
    # 绘制结果
    draw(x_test, decoded, n, 'vae.png')


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--input_shape', default=784, type=int)
    parser.add_argument('--intermediate_dim', default=512, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--latent_dim', default=2, type=int)
    parser.add_argument('--mse', default=False, type=bool)
    params = parser.parse_args()
    main(params)
