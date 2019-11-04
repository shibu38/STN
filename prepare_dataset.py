import os 
import random
from sklearn.model_selection import train_test_split
import pickle
import cv2 
import statistics
import os
import shutil

input_path = "/data/training/prepared_dataset/"
output_path = "/data/training/prepared_dataset/"
if not os.path.exists(output_path):
    os.makedirs(output_path)

num_points = 1

def get_data_info():
    
    total_count = 0 
    min_count=1000000
    for (dirpath, dirnames, filenames) in os.walk(input_path):

        for dirname in dirnames:
            count = len(os.listdir(os.path.join(dirpath, dirname)))
            total_count+=count
            print("Digit: ", dirname, count)
            min_count=min(min_count,count)
    print('Minimum count is ',min_count)
    return min_count

def generate_entire_data():
    
    all_data = {}
    
    height_list = []
    width_list = []
    
    for (dirpath, dirnames, filenames) in os.walk(input_path):
        
        for dirname in dirnames:
            
            files = os.listdir(os.path.join(dirpath, dirname))
            for file in files:   
                         
                if dirname not in all_data:
                    all_data[dirname] = []
                    
                if os.stat(os.path.join(dirpath, dirname, file)).st_size == 0 : 
                    continue
#                 print(os.path.join(dirpath, dirname, file))
                img = cv2.imread(os.path.join(dirpath, dirname, file)) 
                im_height, im_width, _ = img.shape
#                 print(im_height, im_width)
                height_list.append(im_height)
                width_list.append(im_width)
                all_data[dirname].append(file) 

                
    height_list =sorted(height_list)
    width_list =sorted(width_list)    
    print("Median:", statistics.median(height_list), statistics.median(width_list))
    print("Mean:", statistics.mean(height_list), statistics.mean(width_list))
    print("Mode:", statistics.mode(height_list), statistics.mode(width_list))    
    return all_data

def seg_data(data, min_count):
    
    # seg_data =  []
    
    random.seed(a= 10)
    
    for key in data.keys():
        if not os.path.exists(str(key)):
            os.makedirs(os.path.join(output_path,str(key)))
        print(key,len(data[key]))
        if len(data[key])==min_count and False:
            shutil.copy(os.path.join(input_path,str(key)),os.path.join(output_path,str(key)))
        else:
            print('****')
            files=random.sample(data[key],min(len(data[key]),min_count))
            for file_name in files:
                shutil.copyfile(os.path.join(input_path,str(key),file_name),os.path.join(output_path,str(key),file_name))


    #     if len(data[key]) < n :
            
    #         samples = random.sample(data[key], len(data[key]))
            
    #         for sample in samples:
                
    #             data[key].remove(sample)
    #             seg_data.append(sample)
            
    #     else: 
            
    #         samples = random.sample(data[key], n)
            
    #         for sample in samples:
    #             data[key].remove(sample)
    #             seg_data.append(sample)   
                
    # return seg_data, data


min_count=get_data_info()
# data = generate_entire_data()
# seg_data(data,min_count)
# # data_train, data_test = train_test_split(data_lim, test_size=0.10, random_state=42)

# with open('./data/train_class_20_nfl.pkl', 'wb') as fp:
#     pickle.dump(data_train, fp)
    
# with open('./data/test_class_20_nfl.pkl', 'wb') as fp:
#     pickle.dump(data_test, fp)
    
# with open('train_class_20_nfl.txt', 'w') as f:
#     for item in data_train:
#         f.write("%s\n" % item)
        
# with open('test_class_20_nfl.txt', 'w') as f:
#     for item in data_test:
#         f.write("%s\n" % item)