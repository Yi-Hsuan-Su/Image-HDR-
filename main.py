from argparse import ArgumentParser
from hdr import*
from mtb import*


import cv2
import numpy as np
import os
import os.path as osp



class program():

        
        def check_path(self, input_dir, output_dir):
            if not osp.exists(input_dir) or not osp.isdir(input_dir):
                print('[Process] input directory not exists.')
                return False
    
            if not osp.exists(output_dir):
                os.mkdir(output_dir)
    
            return True
    
        def read_images(self, dirname):
            def is_image(filename):
                name, ext = osp.splitext(osp.basename(filename.lower()))
                return ext in ['.jpg', '.JPG', '.png', '.PNG']
    
            images = np.ndarray(shape = (13, 480, 720, 3), dtype = 'float32')
            count=0
            for filename in np.sort(os.listdir(dirname)):
                if is_image(filename):
                    im = cv2.imread(osp.join(dirname, filename))            
                    images[count] = im
                    count+=1
 
    
            print('image shape :', images[0].shape)
            print('images num : = ', len(images))
    
            height, width,channels = images[0].shape
             
            return   height,width  , images
        
        def read_shutter_times(self, dirname):
            shutter_times, shutter_times_string = [], []
    
            with open(osp.join(dirname, 'shutter_times.txt'), 'r') as f:
                
                for line in f.readlines():
                    line = line.replace('\n', '')
                    shutter_times_string += [line]
                    
                    if '/' in line:
                        a, b = np.float32(line.split('/'))
                        shutter_times += [a/b]
                        
                    else:
                        shutter_times += [np.float32(line)]
            
            print('shutter times:', shutter_times_string)
    
            return np.array(shutter_times, dtype=np.float32), shutter_times_string

        def Run_program(self, input_dir, output_dir, 
            tone_map = 'all', 
            n_samples = 100, 
            n_lambda = 30,
            n_depth = 5,
            alpha = 0.5,
            ):
            if not self.check_path(input_dir, output_dir):
                return None
            # 讀黨
            Height, Width, Image = self.read_images(input_dir)
            
            print("H" + str(Height))
            print("W" + str(Width))
            #對齊
            # Image = Image_Alignment().convert2Binary(Image)
    
            # 讀取曝光時間
            st, st_str = self.read_shutter_times(input_dir)
            exp = np.log(st)
            #取樣
            z = HDR().sample_image(Width, Height, Image, n_samples, len(Image))
            #計算Gcurve
  
            rad , hdr = HDR().HDR_Sloution(z, Image, exp, n_lambda ,output_dir)
            
            print("output " +str(hdr.shape)) 

            # convert HDR to LDR
           # ldr = self.tonemapping(rad, tone_map, alpha, output_dir)
            HDR().computeRadianceMap(output_dir, hdr)
            
            
            
        
            
            
if __name__ == '__main__':
    parser = ArgumentParser('High Dynamic Range Imaging')
    parser.add_argument('--input_dir', default='./data/input/04/', type=str, help='輸入路徑')
    parser.add_argument('--output_dir', default='./data/output/04/', type=str, help='輸出路徑')
    parser.add_argument('--tone_map', default='all', choices=['all', 'local', 'global', 'bilateral'], type=str, help='Tone mapping ')
    parser.add_argument('--n', default=100, type=int, help='取樣數')
    parser.add_argument('--lambda_', type=int, default=50 , help ='lambda值')
    parser.add_argument('--d', default=5, type=int, help='MTB 深度')
    parser.add_argument('--a', default=0.5, type=float, help='Alpha for photographic tonemapping')

    args = parser.parse_args()
    # Example Usage
    prg = program()
    prg.Run_program(
        input_dir = args.input_dir,
        output_dir = args.output_dir, 
        tone_map = args.tone_map,
        n_samples = args.n,
        n_lambda = args.lambda_,
        n_depth = args.d,
        alpha = args.a,
        )
    
    print('[Finish]')
    