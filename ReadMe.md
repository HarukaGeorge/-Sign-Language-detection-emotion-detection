#Sign Language Detection And Sentiment Analysis

##Abstract: 

In the United States, around 2 to 3 out of every 1,000 children are born with a detectable amount of hearing loss in one or both ears. Approximately 15% of adults in the United States (37.5 million) say they have difficulty hearing. Communication with hearing impaired (deaf/mute) persons is a major difficulty in today's society; this is due to the fact that their primary mode of communication (Sign Language or local hand gestures) necessitates the use of an interpreter at all times. Non-hearing impaired and hearing impaired people (the deaf/mute) can benefit greatly from circadian engagement with visuals by converting images to text and speech. This gives rise to the need for a medium that can convert sign language to text.




##Process

* Run code DirectoryCreation_DataCollection.ipynb, this will create two directories: dataSet (data collection) and final_dataSet (train test split) and collect the raw images by making the signs in the blue box that appears on screen while using keyboard entry to save the frame to dataSet directory (only if you want to create your own dataset)
* Apply Gaussian Blur to captured raw images and splitting the collected data into train and test folders in the final_dataSet directory by running the GaussianBlur_TrainTestSplit.ipynb (only if you want to create your own dataset)
* Run the Model_Building.ipynb file to train the model and view accuracy/ loss results and saves model into JSON format (model_file.json) and model weights (model_wt.h5) to use in final GUI_SignLanguage.ipynb
* To open the GUI to start sign language translation, run GUI_SignLanguage.ipynb  the  file 
