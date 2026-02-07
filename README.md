This repository contains train.py and predict.py, 2 simple python programs that do the following:

1. train.py trains a model by learning visual patterns from labeled images
2. predict.py uses the trained models to classify a new image, identify the face in the image, and save the result.

In order to run train.py, all folders would first need to be full. In this repository there are some images already present for demonstration purposes.
After running train.py a new file named models.pkl will appear in the folder of the program which contains the trained model on which the prediction will occur.
Once that's done, you may run predict.py which will prompt you to type the full pathname of the image you want to examine. 
After inputting an image, using the above method, it will appear in a new folder named "output", with information printed on the image
