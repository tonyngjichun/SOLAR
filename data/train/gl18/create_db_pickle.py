import os
import csv

import pickle
import numpy as np
np.random.seed(42)

from tqdm import tqdm

def _id_to_qid(_id_list, _id_dict_):
    qid_list = []
    for _id in tqdm(_id_list):
        qid_list.append(_id_dict_[_id])

    return qid_list

csv_fn = 'train.csv'
train_fn = 'boxes_split1.csv'
val_fn = 'boxes_split2.csv'

csvfile = open(csv_fn, 'r')
csvreader = csv.reader(csvfile)
key_landmark_list = [[line[0]] + [line[-1]] for line in csvreader]
all_images_list = [line[0] for line in key_landmark_list]
key_landmark_dict = {line[0]: line[-1] for line in key_landmark_list}

trainfile = open(train_fn, 'r')
trainreader = csv.reader(trainfile)
train_boxes = [line for line in trainreader][1:]

valfile = open(val_fn, 'r')
valreader = csv.reader(valfile)
val_boxes = [line for line in valreader][1:]

train_boxes_dict = {}
for _id, box in train_boxes:
    train_boxes_dict[_id] = box

val_boxes_dict = {}
for _id, box in val_boxes:
    val_boxes_dict[_id] = box
val_list_all = list(val_boxes_dict.keys())

train_list_all = list(set(all_images_list) - set(val_list_all))
train_list = []
val_list = []

landmark_ids = {}
landmark_ids['train'] = []
landmark_ids['val'] = []

train_and_val_list = {}
train_and_val_list['train'] = []
train_and_val_list['val'] = []

print("Finding all training images that succesfully downloaded")
for cid in tqdm(train_list_all):
    if os.path.exists(os.path.join('train', cid + '.jpg')):
        train_list.append(cid)
        landmark_ids['train'].append(key_landmark_dict[cid])
print(len(train_list))
train_and_val_list['train'] = train_list

print("Finding all val images that succesfully downloaded")
for cid in tqdm(val_list_all):
    if os.path.exists(os.path.join('train', cid + '.jpg')):
        val_list.append(cid)
        landmark_ids['val'].append(key_landmark_dict[cid])
print(len(val_list))
train_and_val_list['val'] = val_list

train_list = [cid for cid in train_list_all if os.path.exists(os.path.join('train', cid + '.jpg'))]
val_list = [cid for cid in val_list_all if os.path.exists(os.path.join('train', cid + '.jpg'))]

key_landmark_list = key_landmark_list[1:]  # Chop off header

_id_to_landmark = {}
_id_to_landmark['train'] = {}
_id_to_landmark['val'] = {}

landmark_to_id = {}
landmark_to_id['train'] = {}
landmark_to_id['val'] = {}

db_dict = {}
db_dict['train'] = {}
db_dict['val'] = {}
boxes = {}
boxes['train'] = train_boxes_dict
boxes['val'] = val_boxes_dict


db_dict['train']['cids'] = []
db_dict['train']['qidxs'] = []
db_dict['train']['pidxs'] = []
db_dict['train']['cluster'] = []
db_dict['train']['bbxs'] = []

db_dict['val']['cids'] = []
db_dict['val']['qidxs'] = []
db_dict['val']['pidxs'] = []
db_dict['val']['cluster'] = []
db_dict['val']['bbxs'] = []

_id_dict = {}
_id_dict['train'] = {}
_id_dict['val'] = {}

i = 0

unique_landmarks = {}
unique_landmarks['train'] = list(set(list(landmark_ids['train'])))
unique_landmarks['val'] = list(set(list(landmark_ids['val'])))


landmark_to_qids = {}
landmark_to_qids['train'] = {}
landmark_to_qids['val'] = {}

landmarks_path = 'landmark_to_qids.pkl'

# finding idxs that corresponds to each landmark
print('finding idxs that corresponds to each landmark...')
for mode in ['train', 'val']:
  for landmark in tqdm(unique_landmarks[mode]):
      landmark_to_qids[mode][landmark] = np.where(np.array(landmark_ids[mode]) == landmark)[0].tolist()

for mode in ['train', 'val']:
    image_list = train_list if mode == 'train' else val_list
  
    boxes_dict = boxes[mode]

    for i, image in enumerate(tqdm(image_list)):
        positives = []

        db_dict[mode]['cids'].append(image)

        landmark = key_landmark_dict[image]
        db_dict[mode]['cluster'].append(landmark)

        pidxs_potential = landmark_to_qids[mode][landmark]


        try: 
            pidxs_potential.remove(image)
        except:
            pass

        try:
            pidxs_potential.remove(i)
        except:
            pass

        if len(pidxs_potential) == 0:
            continue
    
        pidxs = np.random.choice(pidxs_potential, min(len(pidxs_potential), 1)).tolist()

        try:
            bbx = list(map(float, train_boxes_dict[image].split()))
            db_dict[mode]['bbxs'].append(bbx)
        except:
            db_dict[mode]['bbxs'].append(None)

        db_dict[mode]['qidxs'].append(i)
        db_dict[mode]['pidxs'].append(pidxs)


save_path = './db_gl18.pkl'

pickle.dump(db_dict, open(save_path, 'wb'))
