# lsf_subtitlehttp://9gag.com/
Automatic subtitle of Sign Language using FFNN and Mediapipe


## How to use :

This project uses Python 3.
Required packages are mediapipe, cv2, sklearn (and also numpy, csv, os, time, uuid and sys but these are most likely to be already installed in your environment)

### First step
You can easily make out your own base of signs using the "enhanced_vocab_making.py" file. You will have to acquaint the list of words you are setting up in the vocabulary, and the amount of images you are ready to take to train the neural network. The needed data will be loaded in 'mp_data.csv'.

You can also use our prerecorded data, in the file 'mp_data_base.csv' and skip this step.

### Second step
Then, you will have to run the 'fast_forward_neural_network.py' file, and acquaint again the same list of words and number of images you gave on first step.

If you are using our prerecorded data, you can also skip this step.

### Third step
Finally, you can use the neural network to subtitle sign language videos or camera input :

For video input : run 'name.py' and acquaint the link of the video.

For camera input : run 'name2.py' and sign to your camera.
