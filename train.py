from dataset import dataGenerator
from arg import args_infom,easy_arg
import numpy as np
from utill import upsample,downsample,downsample_2,image_to_patches,images_to_patches,restore_patches_to_image
from noise import add_gaussian_noise
from network import FfdNet
import cv2
import os
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

args_infom()                    #打印数据并保存在本地的一个js文件中
arg = easy_arg()                #使用python中的保存的数据

def main():
    generator = FfdNet(layer=12) # 生成器
    # x = np.random.random(size=(1024,16,16,12))
    # xx = np.random.random(size=(1024,16,16,3))
    # generator(x,xx)
    generator.load_weights("./ckpt/best_d.hd5")
    best_weights_checkpoint_path_g = os.path.join(arg.ckpt_path, 'best_d.hd5')
    best_weights_checkpoint_g = ModelCheckpoint(best_weights_checkpoint_path_g,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True,
                                                mode='min')

    IMG = dataGenerator(folder=r"4", im_size=1024, flip=True)

    noise_scal = arg.train_noise_intercal  # 获取噪声
    noise_scal = np.array([i for i in np.arange(noise_scal[0], noise_scal[1], noise_scal[2])]) / 255.0  # 将噪声标准化
    summary_writer = tf.summary.create_file_writer(arg.log)

    for epoch in range(arg.epochs):

        img = IMG.get_batch(num=arg.batch)                   #读取数据
        img_patch = images_to_patches(img,arg.patch_size)
        img_real = downsample(img_patch)
        noise_scal_ = noise_scal[np.random.randint(low=0,high=len(noise_scal))]         #随机获得一个噪声参数
        img_noise = add_gaussian_noise(img_patch,variance=noise_scal_)             #添加噪声后的图片
        img_noise_ = restore_patches_to_image(img_noise,patch_size=arg.patch_size)
        img_noise_input = downsample(img_noise)
        noise_scal_img = np.broadcast_to(noise_scal_.reshape((1,1,1,1)),(img_noise_input.shape[0],img_noise_input.shape[1],img_noise_input.shape[1],3))
        with tf.GradientTape() as tape:
            predict_img = generator(img_noise_input,noise_scal_img)
            loss = tf.reduce_mean(tf.square(img_real - predict_img))
        g_grads = tape.gradient(loss, generator.trainable_variables)
        tf.optimizers.Adam(learning_rate=arg.lr).apply_gradients(zip(g_grads, generator.trainable_variables))

        Predict_Img = upsample(predict_img)
        Img = restore_patches_to_image(Predict_Img,arg.patch_size)

        with summary_writer.as_default():
            tf.summary.scalar('loss', float(loss), step=epoch)
            if epoch % 100== 0:
                tf.summary.image("fake_image", np.expand_dims(Img,0), step=epoch)
                tf.summary.image("real_image", np.expand_dims(img[0],0), step=epoch)
                tf.summary.image("noise_image",np.expand_dims(img_noise_,0), step=epoch)
                tf.keras.backend.clear_session()
                print("tf.keras.backend.clear_session")
                generator.save_weights(best_weights_checkpoint_path_g)
                print(epoch, "loss:", float(loss))
    return None
if __name__ == '__main__':
    main()