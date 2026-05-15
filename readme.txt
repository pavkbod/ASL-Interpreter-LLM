How code is structured: 

- Code is structured by three seperate files: capture_handsigns.py, which basically is a 
script that takes pictures of your handsigns and puts them into the organized dataset, 
asl_classifier.py which handles training pipeline, and trains neural network, and webcam.py which loads the trained model
and predicts the handsign you do on camera. Each file is structured mainly using def helper functions and classes. 

- There is also asl_model.npz. This is generated when running asl_classifier.py. It is the saved trained model
which contains learned weights, biases, and class names.

- There is also the dataset including already 500 pictures of each lettes: A, B, C, L, Y, and the phrase, I love you.


Instructions to run code: 

    - If you haven't already installed the libraries, please do so using:
    python3 -m pip install numpy opencv-python. You can check if you already
    have the libraries by doing: python3 -m pip show numpy, and python3 -m pip show opencv-python

    - Then if you want to add your own handsigns or more data to the already existing handsigns 
    to the database, you do python3 capture_handsigns.py
    When running this you will get the prompt: "Enter the sign label (example: A, B, C, L, Y, I_love_you):"
    Enter your desired label, press enter, and then the webcam will appear with a green square. Put your handsigns
    in the green square, and press t if you want to do a test shot, or s to do 100 shots of your desired handsign. 
    REMEMBER TO DO STATIC HANDSIGNS!!! If you plan on doing a new handsign, I heavily reccomend to do so in a well
    lit room, and do at least 500 pictures, with some variation using both hands. 

    - To train the model, do python3 asl_classifier.py which will train the model and then save the trained model to
    asl_model.npz, your output will show something like this
    
    EXAMPLE OUTPUT:

    Epoch 1/50 | Loss: 1.8462 | Train Accuracy: 0.1997
    Epoch 5/50 | Loss: 1.7666 | Train Accuracy: 0.2737
    Epoch 10/50 | Loss: 1.7103 | Train Accuracy: 0.2887
    Epoch 15/50 | Loss: 1.6701 | Train Accuracy: 0.4605
    Epoch 20/50 | Loss: 1.6422 | Train Accuracy: 0.3409
    Epoch 25/50 | Loss: 1.6175 | Train Accuracy: 0.4894
    Epoch 30/50 | Loss: 1.5757 | Train Accuracy: 0.3965
    Epoch 35/50 | Loss: 1.5452 | Train Accuracy: 0.5207
    Epoch 40/50 | Loss: 1.5056 | Train Accuracy: 0.4685
    Epoch 45/50 | Loss: 1.4725 | Train Accuracy: 0.5697
    Epoch 50/50 | Loss: 1.4360 | Train Accuracy: 0.5120

    Test Accuracy: 0.5661

    Confusion Matrix:
    True\Pred      A              B              C              I_love_you     L              Y              
    A              95             0              1              5              1              0              
    B              12             88             0              0              34             0              
    C              0              0              43             2              73             0              
    I_love_you     26             0              0              55             25             0              
    L              2              0              0              1              121            0              
    Y              32             1              0              59             38             5              

    Per-Class Accuracy:
    A: 0.9314 (95/102)
    B: 0.6567 (88/134)
    C: 0.3644 (43/118)
    I_love_you: 0.5189 (55/106)
    L: 0.9758 (121/124)
    Y: 0.0370 (5/135)
    Model saved to asl_model.npz


    - Lastly, run python3 webcam.py and your camera will load the model, and open your camera with the 
    same green square and will predict the handsign you do! Make sure to do this in a well lit 
    room and have a non-blurry camera. Have fun!





