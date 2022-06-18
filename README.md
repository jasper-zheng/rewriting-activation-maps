# Rewriting Activation Maps

## Summary  
This project proposed a novel training system appended to StyleGAN2 architecture, enabling a pre-trained StyleGAN2 model to perform image-to-image translation, even if the input images are not in the original domain. The training system is based on an encoder network that downscales the generated images from a StyleGAN2 model and matches the distribution of the earlier feature maps in the same model. After training, the encoder network is migrated to the StyleGAN2 model.   

The proposed system was implemented on a couple of pre-trained models. And the results showed that it's able to create meaningful image-to-image translation similar to pix2pix and other state-of-the-art image translation models.  

In addition, a real-time interactive system was built to facilitate human control of the network.   

## Related Study and Motivation  

In addition to improving the model performance and quality, modern approaches also focus on manipulating the trained network to produce outputs that are diverse from the original dataset.  

Model Rewriting showed that editing a network's internal rules allows us to map new elements to the generated image intentionally.

Network Bending showed that the transformations of spatial activation maps in GANs could create meaningful manipulations in the generated images. 

Both of these works indicated that the knowledge encoded in a deep neural network is semantically related to the spatial information in its feature maps. And manipulating this information can create results diverse from the original domain. Therefore, we asked:  
 * Could we introduce an additional network to learn the spatial distribution of information in a specific layer in a trained GAN model? 
 * And with this additional network, if we could directly generate feature maps from a given real image and create image-to-image translation?

## Further Study  

The current stage of the project only tested the proposed training system on limited layers (i.e. after the 16x16 synthesise block) in a few trained models (i.e. [Frea Buckler artwork](https://twitter.com/dvsch/status/1255885874560225284), [MetFaces](https://twitter.com/ak92501/status/1282466682267676675), [FFHQ](https://github.com/NVlabs/ffhq-dataset), [A Database of Leaf Images](https://data.mendeley.com/datasets/hb74ynkjcn/1)). Future studies might scale up the experiments on different layers in different networks. They might also focus on testing the training system with different settings, or inserting jittering layers and mean filter layers to improve the output quality.  

The StyleGAN2 implementation borrowed heavily from [moono/stylegan2-tf-2.x](https://github.com/moono/stylegan2-tf-2.x)  
