import os

with open("ccpd_net_list.txt", 'x') as list_file:
    for i in range(10000):
        list_file.write(f"/mnt/hgfs/ubuntu/ccpd/val/images_bgr/{i+1}.bgr\n")