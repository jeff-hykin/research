1) Did the authors make their codes publicly available?
    Sadly, I believe they did not. The paper doesn't say, and after searching online there doesn't seem to be any repositories closely related to the paper.
2) Is it clear how we can reproduce their method? If yes, great. If not, how can we decompose it into sub-tasks, which sub-tasks are easy to do, and which sub-tasks are harder?

If the goal is to create something similar with similar performance I would say that yes it is clear, and there is only a few isolated things I need clarified. For example, I don't know how to exactly replicate their augmentations, but I do know how to apply relevant/similar augmentations for this kind of data.

If the goal is to perfectly reproduce the paper (for scientific reproducability and statistical power), then I would say no, there's several details I would need to clarify.

For a pratical application, I'm only unclear about:
1. Some of the mathematical notation in the loss function
2. Some of the mathematical notation in the weighted sum function
3. if only the sub-networks are trained (independently), or if the sub-networks are trained, and then there is a seperate session for training the full ensemble

For scientific reproducibility I have broken down the paper into a sub-tasks.
The challenging steps to me are:
- using/learning the Dlib C++ library
- using/learning the Caffe library
- understanding how to transfer the pretraining of VGG to the new fully connected layers
- understanding the confusion matrix; how it is produced and what it means pratically
- understanding the exact augmentations

Pre-Knowledge I needed (and had to lookup):
- the AlexNet
    - see https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637
    - layers:
        - (3, 227, 227) Input
        - (96, 55, 55 ) Conv1: Relu 
        - (96, 27, 27 ) Max Pooling
        - (256, 27, 27) Norm, Conv2: Relu
        - (256, 13, 13) Max Pooling
        - (384, 13, 13) Norm, Conv3: Relu
        - (384, 13, 13) Conv4: Relu
        - (256, 13, 13) Conv5: Relu
        - (256, 6, 6  ) Max Pooling
        - 4096 Dropout: rate=0.5, FC6: Relu
        - 4096 Dropout: rate=0.5, FC7: Relu
        - 1000 FC8: Relu
                
- the VGG-16 network:
    - link https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/ 
    - the link for the channel depths, 
    - kernel size of 3x3
    - all of the layers (except the last one) use relu activation
    - layers:
        - 224 convolutional layer 
        - 224 convolutional layer 
        - 112 max pooling layer
        - 112 convolutional layer 
        - 112 convolutional layer 
        - 56 max pooling layer
        - 56 convolutional layer 
        - 56 convolutional layer 
        - 56 convolutional layer 
        - 28 max pooling layer
        - 28 convolutional layer 
        - 28 convolutional layer 
        - 28 convolutional layer 
        - 14 max pooling layer
        - 14 convolutional layer 
        - 14 convolutional layer 
        - 14 convolutional layer 
        - 7 max pooling layer
        - 4096 fully connected layer
        - 4096 fully connected layer
        - 1000 fully connected layer
        - a softmax activation

Sub-Tasks:
1. Download
- Real-world Affective Faces Database RAF-DB
- Acted Facial Expressions in the Wild AFEW 7.0
2. Process the data
- Ambiguous: for AFEW, filter out any frames with unclear faces
- Ambiguous: Sample the clear faces at 3-10 fps with an adaptive interval
- for all remaining images (in both RAF-DB and AFEW)
    - flip the image (horizontally?)
    - Ambiguous: (if from the RAF?) rotate it (uniform-random-distribution?) between ±4°
    - Ambiguous: (if from the AFEW?) rotate it (uniform-random-distribution?) between ±6°
    - then add Gausian whie noise with a variance of 0.001
    - then add Gausian whie noise with a variance of 0.01
    - then add Gausian whie noise with a variance of 0.015
- there are 95465 cropped face images from the RAF-DB data
- for each facial picture
    - use Dlib C++ library to locate 68 facial landmarks
    - crop the
        - whole face
        - left eye
        - nose
        - mouth
    - force each cropping into a 224x224 image
    - create three pairs of images
        - (whole face, left eye)
        - (whole face, nose)
        - (whole face, mouth)
    - all 3 of these pairs form a "sample"
3. Sub network creatation
- use the Caffe framework on Ubuntu
- Create a "vgg-sub-network" by doing the following:
    - Knowledge limitation: does the VGG-16 model come pre-trained?
    - take the standard pre-trained VGG-16 network
        - remove the fully connected layers
        - duplicate it into two networks, a "FullFace" network and a "PartialFace"  each have 7x7 output
    - take the 7x7 output of network "FullFace" and 7x7 output "PartialFace" and concatenate them into a 7x14 dimension matrix
    - Question: why are the fully connected layers called "fc6", "fc7", "fc8" if there's only three of them
    - connect the 7x14 output to a 1x4096 fully connected layer
    - connect that output to a 1x4096  fully connected layer
    - connect that output to a 1x7 fully connected layer
    - Ambiguous: are the fully connected layers supposed to retain any of the pretraining from the vgg-16 model
    - use softmax to get final 1x7 output
    - use relu activation for all of the other layers
- Create a "alex-sub-network" by doing the following:
    - take the standard non-pre-trained AlexNet
        - change the 2nd to last layer (fully connected) to have 64 dimension output
        - change the last layer to have 7 dimension output
        - duplicate it into two networks, a "FullFace" network and a "PartialFace" both each have 7x7 output
    - repeat the same steps as the "vgg-sub-network" for combining the two 7x7 outputs
4. Full network creation
- to create the full vgg network
    - Duplicate the "vgg-sub-network" 3 times
    - name them "nose-sub-net", "eye-sub-net", and "mouth-sub-net"
    - for each sample  inthe dataset, feed the respective (nose/eye/mouth) image pair into the sub-net
    - take the output vector (7 dimensions) of each of the sub networks
    - combine them using the weighted_sum_operation()
- to create the full alex network
    - repeat the same steps as above, but replace "vgg-sub-network" with "alex-sub-network"
5. Training process
- knowledge limitation: I don't understand some of the mathamatical notation of the weighted_sum() operation
- knowledge limitation: I don't understand some of the mathamatical notation of the loss() function
- given one of the datasets
- first train the vgg-sub-networks
    - 20k iterations
    - batch size = 16
    - Ambiguous: learning rate is 0.0001-0.0005
    - weight decay = 0.0001
    - momentum = 0.9
    - use linear learning rate decay in the stochastic gradient decent (SGD) optimizer
    - use the loss() function defined in the paper
- then the alex-sub-networks
    - 30k iterations
    - batch size = 64
    - learning rate starts at 0.001
    - use the loss() function defined in the paper
- Ambiguous: it is unclear to how the full network is trained: whether the sub networks are the only thing that are trained or if there is a seperate training of the ensemble as a whole
- knowledge limitation: somehow the following paramters are applied to the weighted_sum_operation() and the loss() function
    - Full VGG Network
        - 4/7 (left-eye weight)
        - 2/7 (mouth weight)
        - 1/7 (nose weight)
    - Full Alex Network
        - 2/5 (left-eye weight)
        - 2/5 (mouth weight)
        - 1/5 (nose weight)
6. RAF-DB validation
- perform 5 fold cross validation using the (whole? training+test?) augmented RAF-DB dataset
7. AFEW validation
- perform standard validation on the frames with faces in the validation set of the AFEW dataset