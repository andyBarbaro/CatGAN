# CatGAN
This project generates fake 64x64 pixel images of cats' faces using a generative adversarial network (GAN). The generator and discriminator networks were developed using the Pytorch library. A convolutional neural network is used for the discriminator and transposed convolutional is used for the generator. Each epoch of training is expected to take around 20 minutes.

Included in this repository is a folder containing approximately 16,000 images of cat faces used for input data. Additionally, I have included a sample of 64 cat faces developed after each epoch of training the model. These 64 cat faces are based on the same random noise input after each epoch of training. The grid of images is displayed in a short video in order to more clearly show the improvement of the model while training. Lastly, a paper written about GANs, my model, and some potential future improvements is included for those interested in further research.
