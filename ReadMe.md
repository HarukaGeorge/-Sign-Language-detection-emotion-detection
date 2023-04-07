# **Sign Language Detection And Sentiment Analysis**
---

## _Abstract_: 

In the United States, around 2 to 3 out of every 1,000 children are born with a detectable amount of hearing loss in one or both ears. Approximately 15% of adults in the United States (37.5 million) say they have difficulty hearing. Communication with hearing impaired (deaf/mute) persons is a major difficulty in today's society; this is due to the fact that their primary mode of communication (Sign Language or local hand gestures) necessitates the use of an interpreter at all times. Non-hearing impaired and hearing impaired people (the deaf/mute) can benefit greatly from circadian engagement with visuals by converting images to text and speech. This gives rise to the need for a medium that can convert sign language to text.

## Dataset Collection and Description

We're creating our own dataset because our research is built on a vision-based approach that necessitates the use of raw images. The most generally available dataset for sign language problems is in RGB format, which is incompatible with our needs. As a result, we decided to gather our own data.

![Format of Data](https://github.com/HarukaGeorge/Sign-Language-detection-sentiment-analysis/blob/master/images/data_format.jpg)

![Data Collection](https://github.com/HarukaGeorge/Sign-Language-detection-sentiment-analysis/blob/master/images/data_collection.jpg)

In our original dataset of raw images we apply the Gaussian Blur Filter to the images, which aids in the extraction of numerous features. A Gaussian Filter is a low-pass filter that is used to reduce noise (high-frequency components) and blur picture regions.

A glimpse of dataset: 

![Glimpse of data](https://github.com/HarukaGeorge/Sign-Language-detection-sentiment-analysis/blob/master/images/data_glimpse.jpg)

## Process

* Run code DirectoryCreation_DataCollection.ipynb, this will create two directories: dataSet (data collection) and final_dataSet (train test split) and collect the raw images by making the signs in the blue box that appears on screen while using keyboard entry to save the frame to dataSet directory (only if you want to create your own dataset)
* Apply Gaussian Blur to captured raw images and splitting the collected data into train and test folders in the final_dataSet directory by running the GaussianBlur_TrainTestSplit.ipynb (only if you want to create your own dataset)
* Run the Model_Building.ipynb file to train the model and view accuracy/ loss results and saves model into JSON format (model_file.json) and model weights (model_wt.h5) to use in final GUI_SignLanguage.ipynb
* To open the GUI to start sign language translation, run GUI_SignLanguage.ipynb  the  file 

## Outcomes and Findings
- The model accurately identify the majority of alphabets and display the words and sentences that these alphabets form. Our algorithm also examines the sentiment of the currently predicted word and returns a negative or positive result, along with any other emotions indicated by the word
- The model can be enhanced by adding the functionality of predicting the sentiment for entire sentences, it is one of the future scopes of the project
- Furthermore, even in the case of complex backgrounds, greater accuracy can be attained by experimenting with different background subtraction techniques
- For the model, a web or mobile application can be developed so that it is easily accessible regardless of device type or location

## Demos

Detection of word "CAT"

<img src="https://github.com/HarukaGeorge/Sign-Language-detection-sentiment-analysis/blob/master/images/demo1.jpg" width="240" height="300" /> <img src="https://github.com/HarukaGeorge/Sign-Language-detection-sentiment-analysis/blob/master/images/demo2.jpg" width="240" height="300" /> <img src="https://github.com/HarukaGeorge/Sign-Language-detection-sentiment-analysis/blob/master/images/demo3.jpg" width="240" height="300" /> <img src="https://github.com/HarukaGeorge/Sign-Language-detection-sentiment-analysis/blob/master/images/demo4.jpg" width="240" height="300" /> 



Detecting sentence "I am good" and showing sentiment "Positive"
![Demo6](https://github.com/HarukaGeorge/Sign-Language-detection-sentiment-analysis/blob/master/images/demo6.png)

