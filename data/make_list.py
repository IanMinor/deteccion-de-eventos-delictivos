import numpy as np
import os
import glob

def make_train_list(rooth_path, output_file):
    dirs = os.listdir(rooth_path)
    print("Carpetas train encontradas: ", dirs)
    
    with open(output_file, 'w') as f:
        normal = []
        for dir in dirs:
            subdir_path = os.path.join(rooth_path, dir, dir)
            if os.path.exists(subdir_path):
                files = sorted(glob.glob(os.path.join(subdir_path, '*.npy')))
            else:
                files = sorted(glob.glob(os.path.join(rooth_path, dir, '*.npy')))
            
            for file in files:
                if not '__' in os.path.basename(file):
                    normalized_path = file.replace('\\', '/').replace('../', '')
                    if 'Normal_' in file:
                        normal.append(normalized_path)
                    else:
                        f.write(normalized_path + '\n')
        
        print(f"Videos anómalos: {f.tell()//80} aprox, Videos normales: {len(normal)}")        
        for file in normal:
            f.write(file + '\n')
            
def make_test_list(rooth_path, output_file):
    print('Generando lista de test...')
    with open(output_file, 'w+') as f:
        files = sorted(glob.glob(os.path.join(rooth_path, '*/*.npy')))
        for file in files:
            normalized_path = file.replace('\\', '/').replace('../', '')
            f.write(normalized_path + '\n')

if __name__ == '__main__':
    train_path = '../I3D_features/Train/RGB'
    test_path = '../I3D_features/Test/RGB'
    
    # Archivos de salida
    train_list = 'ucf-i3d.train.list'
    test_list = 'ucf-i3d.test.list'

    make_train_list(train_path, train_list)
    make_test_list(test_path, test_list)
