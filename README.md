## Automatic Lung Nodule Detection based unsupervised learning using PET/CT images

### Overview
This project is a tensorflow and Keras implementation, 
which performs lung nodules detection based unsupervised learning using PET/CT images. 
The main idea for this is that better image representation gets better clustering, 
and better clustering results helps to get better image representation. 
This model is based on unsupervised learning so it can use raw DICOM data from medical instruments. 

![10  NoduleDetection](https://user-images.githubusercontent.com/52024566/73548729-9b4b5a00-4484-11ea-8b69-8c5410ac8b00.png)

### License
This code is released under the MIT License.

### Package Dependencies
1. Pydicom(https://pydicom.github.io/).  
It is used to convert DICOM image to 8-bit image.  
Install pydicom by:  
    ```bash
    $ pip install -U pydicom
    ```
    or you can install pydicom using conda:
    ```bash
   $ conda install pydicom --channel condal-forge
    ```
2. Pandas(https://pandas.pydata.org/).  
It is used to make dataframe for lung image extraction and result analysis.  
Install pandas by:
    ```bash
   $ pip install pandas
    ```

3. Tensorflow 1.14.0(https://www.tensorflow.org/).  
It is used to train network.  
Install tensorflow 1.14.0 by:
    ```bash
   $ pip install tensorflow-gpu==1.14.0 
    ```
   
4. OpenCV 4.1.2(https://opencv.org/).  
It is used for postprocessing.  
Install opencv 4.1.2 by:
    ```bash
   $ pip install opencv-python==4.1.2
    ```
   
5. scikit-image 0.16.2(https://scikit-image.org/).
It is used to import multi-otsu thresholding.  
Install scikit-image 0.16.2 by:
    ```bash
   $ pip install scikit-image
    ```

### Process Overview

![12  ProcessOverview](https://user-images.githubusercontent.com/52024566/73510461-87bcd680-4425-11ea-8548-fa10056ba93a.png)

1. Preprocessing
    * Extract whole slices from PET/CT DICOM 
        ```bash
       $ python python SliceExtraction.py (path_src)
        ```
      
    * Extract lung slices only for training/test
        ```bash
        $ python SliceSelection.py (path_csv, path_src, path_dst)
        ```
        You should make csv files which has information about the location of lung slices.  
        It should have three columns, 'Patient', 'Up', and 'Down'.  
        * Patient : Folder name for each patients  
        * Up : First slice location for extraction  
        * Down : Last slice location for extraction
2. Training Networks
    ```bash
   $ python Training.py
    ```
   Revise FIXME part of this code to set your hyperparameter and configurations.
3. Evaluation  
    This part predicts labels for patches divided from original test image.
    ```bash
   $ python Evaluate.py (path_model, path_image, ind_CT, ind_PT, num_labels)
    ```
   You don't need to revise this code for set your configurations.  
   Please read README for Evaluate folder.
4. Postprocessing  
    This part makes the nodule detection image using predicted labels.
    ```bash
   $ python Postprocessing.py (path_result)
    ```
5. Analysis  
    This part makes the confusion matrix using predicted image and ground truth image.
    ```bash
   $ python Calculate.py (path_result, path_ref)
    ```
    You should make index.csv in your root directory for detection results.
    It should have three columns, 'patient_name', 'feature_index', and 'offset'.  
    * patient_name : Folder name for each patients  
    * feature_index : Feature index for calculate confusion matrix  
    * offset : Offset for detection image and ground truth
    
### Compared Approaches
Two approaches were committed to evaluate the performance of this project.  
1. Convolutional Autoencoder (CAE)  
    CAE is widely used for image segmentation problem. 
    You can see network structure reading README in Network folder.
    Also You can test this model using such files in this project.
    ``` text
       Training_AE.py, Evaluate_AE.py
    ``` 
2. Joint unsupervised learning with single modality (CT)  
    This model was executed to compare the performance difference between single modality and dual modality. 
    You can test this model using such files in this project.
    ``` text
        Training_single.py, Evaluate_single.py
    ```
### Q&A
You are welcome to send message to (melt.ep00 at gmail.com) if you have any issue on this code.
### Reference
1. T. Moriya et al., “Unsupervised segmentation of 3D medical images based on clustering and deep representation learning,” 2018, p. 71.
2. https://github.com/jwyang/JULE.torch
3. https://github.com/AdrianUng/keras-triplet-loss-mnist
 
