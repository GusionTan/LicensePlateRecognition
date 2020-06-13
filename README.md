# LicensePlateRecognition
This is a license plate recognition system based on deep learning, mainly used to recognize Chinese license plates.

## Training
Create new cnn_char_test and cnn_plate_test under carIdentityData to add test pictures.

Change the train_flag of charNeuralNet.py in the main function to 1, execute the program to train the character recognition model, and the model is saved in the carIdentityData/model/char_recongnize path.

Change the train_flag of charNeuralNet.py in the main function to 0, modify the model_path to the path of your own model, and execute the program to test the character recognition accuracy.

The operation steps of plateNeuralNet.py are the same as the two steps above, used to obtain the license plate detection model;



## Running
Modify the plate_model_path and char_model_path of the main.py program to the model path you trained.

`./main.py`
