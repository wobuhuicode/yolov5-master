import os

with open("ccpd_letterbox_list.txt", 'x') as list_file:
    for i in range(10000):
        list_file.write(f"/root/nfs/ccpd/val/images_letterbox/{i+1}.jpg\n")