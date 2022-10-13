import glob
import os
import shutil

Dir = '/media/mcav/One Touch/Datasets/SV2TTS/encoder'  #Source directory containing encoder preprocessed files 
OUTPUTFILE = '/media/mcav/One Touch/folder.txt' #.txt destination directory

# Dir = '/media/mcav/One Touch/Test'
# OUTPUTFILE = '/media/mcav/One Touch/folder.txt'

path = os.path.join(Dir, '*')
for folder in glob.glob(path):
    with open(OUTPUTFILE, 'a') as f:
        if len(os.listdir(folder)) < 2:
            f.write("{0} is a directory with No files'\n' ".format(folder))
            shutil.rmtree(folder)