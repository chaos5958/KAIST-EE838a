#Dependency
- python 3.6
- tensorflow-gpu 1.12

#Train(GPU is required)
1) Prepare train, validation data as follows
- data/[train, valid]/[HR, LR]/*.jpg
2) Execute generate_tfrecord.py
3) Execute train.py

#Test(GPU is required)
1) Prepare test data as follows 
- data/example/LR/*.jpg 
2) Execute test.py 
3) Ouput images are saved to data/example/HR/*.jpg
