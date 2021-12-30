import glob
import os


def read_paths(data_folder, include_negatives=False):
    """
    Read the file paths that are in the subfolders inside data_folder.

    :param data_folder: folder containg the subfolders with images
    :param include_negatives: whether to include additional negatives or not
    :return: directory of classes with image paths inside the classes
    """
    # get the subfolders
    class_paths = glob.glob(os.path.join(data_folder, '*'))
    class_paths = [class_path for class_path in class_paths if os.path.isdir(class_path)]
    if not include_negatives:
        class_paths = [class_path for class_path in class_paths if 'egyik_sem' not in class_path]

    # get all the labels and the datapaths
    return {class_path.split('/')[-1]: read_image_paths_in_folder(class_path) for class_path in class_paths}


def read_image_paths_in_folder(folder):
    """
    Read the paths of all the images in a folder.
    Images should be .jpg or .png .

    :param folder: folder path to look for images
    :return: the paths of all the images
    """
    return glob.glob(os.path.join(folder, '*.jpg')) + \
           glob.glob(os.path.join(folder, '*.png')) + \
           glob.glob(os.path.join(folder, '*.JPG'))
