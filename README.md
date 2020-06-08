# Mask-Detection
Detects if a person is wearing a mask or not through our webcam or any other device.Proper lighting condition are needed for the model to give better and accurate results.I implemented the entire model using CNNs.Having trained it for over 50epochs,I received an accuracy of 95.6%
Tried to tune the hyperparameters using various methods.The best ones were used for the final training.
Although its completely up to the users to use them in their own preferable way.
The link to the Mask dataset is here : https://github.com/prajnasb/observations/tree/master/experiements/data
Since only about 1.5k images are given, try using augmentation for better results.
The file webcam is where you load the pretrained model(mask_trained.h5).Its the file that acceses your webcam.
Be sure to set the value of VideoCapture(x) to 1 or 0 as per your camera.
I used my phone camera, hence I set it to 1.For webcam it should be set to 0.
## Happy HACKING
