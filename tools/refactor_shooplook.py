import os
from shutil import copyfile

def main():
    datapath = '../data/imgs_top/'
    folder_dest = '../data/refactored_imgs_top/'
    clothes = os.listdir(datapath)
    for root in clothes:
        root_path = datapath + root + '/'
        styles = os.listdir(root_path)
        if styles:
            main_image = None
            for file in styles:
                if '.jp' in file:
                    main_image = file
                    styles.remove(file)
                elif '.' in file:
                    styles.remove(file)
            if main_image:
                for style in styles:
                    items_path = root_path+style
                    items = os.listdir(items_path)
                    if items:
                        newpath = folder_dest + root + '_' + style
                        print(newpath)
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)

                        copyfile(root_path + main_image, newpath + '/' + main_image)
                        for item in items:
                            if '.jp' in item:
                                new_name = main_image.split('.')[0] + '_' + item.split('.')[0] + '.' + item.split('.')[1]
                                copyfile(items_path + '/' + item, newpath + '/' + new_name)



if __name__ == "__main__":
    main()