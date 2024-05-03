import os
import pickle
from shutil import copyfile

query_path="/media/nyma/0e9cb3b9-b422-4047-a34b-8ecacb40dea7/Nyma/VRAI-20210507T220717Z-001/VRAI/images_dev/"
query_save_path="/media/nyma/0e9cb3b9-b422-4047-a34b-8ecacb40dea7/Nyma/VRAI-20210507T220717Z-001/VRAI/query/"

dataset_path = os.path.join('/media/nyma/0e9cb3b9-b422-4047-a34b-8ecacb40dea7/Nyma/VRAI-20210507T220717Z-001/VRAI/test_dev_annotation.pkl')
with open(dataset_path, 'rb') as handle:
    data= pickle.load(handle)
    # print(data)

val_list = list(data.values())[0]
val_list1 = list(data.values())[9]
# val_list2 = list(data.values())[7]
# print(len(val_list))
print(len(val_list))
print(len(val_list1))
# print(len(val_list2))
for i in range(0,len(val_list)):
    name=(val_list[i])

    # ID = val_list[i].split('_')
    src_path = query_path + name
    # print(src_path)
    #     continue
    #     print("Here")
    # dst_path = query_save_path + ID[0]
    dst_path=query_save_path

    if not os.path.isfile(src_path):
      continue
    copyfile(src_path, dst_path + '/' + name)
    # os.remove(src_path)
    # else:
    #     copyfile(src_path, dst_path + '/' + name)
