
with open("val_list.txt", 'x') as file:
    for i in range(10000):
        file.write("/root/nfs/bdd-expr-on-board/bdd_val/images/" + str(i + 1) + ".jpg\n")  