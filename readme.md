#Variational Auto-Encoders for texts and images
This repository contains pytorch implementation of various VAEs, working for text (SST-2 at the moment) and images (MNIST at the moment). You can see the config fils in Configs and run any model with python main.py --config_file PATH_TO_FILE.

##Implemented models
###AE
An auto-encoder, useful for comparisons

###VAE
The classic variational auto-encoder, with encoder, decoder, KL divergence, and that's it. 

###CVAE- Classic
TODO

###CVAE
VAE where some dimensions of the latent space are taken to force some attribute, ex: the class.