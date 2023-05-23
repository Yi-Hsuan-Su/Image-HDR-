import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


class HDR():
         
    def sample_image( self,Height,Width, Img , SampleNum , ImageNum):
        rdx_x = list(range(Width))
        rdx_x = random.sample(rdx_x ,SampleNum)
        rdx_y = list(range(Height))
        rdx_y = random.sample(rdx_y ,SampleNum)
        
        z = np.ndarray(shape = (3 , SampleNum , ImageNum) ,  dtype ='uint8')
        
        for i in range(0 , SampleNum): 
            for j in range(0 , ImageNum):
                z[0][i][j] = Img[j][rdx_x[i]][rdx_y[i]][0] #Red
                z[1][i][j] = Img[j][rdx_x[i]][rdx_y[i]][1] #Green
                z[2][i][j] = Img[j][rdx_x[i]][rdx_y[i]][2] #Blue
    
        return z
    
    def compute_weight(self,z):
        zmax = 255.
        zmin = 0.
        zrange = zmax + zmin
        if z<=(zrange/2):
            return z-zmin
        else:
            return zmax-z
    
    def generate_curve(self ,z, exp ,l):
        n = 255
        
        A = np.zeros(shape= ((z.shape[0]*z.shape[1]+n ,z.shape[0]+n+1)) , dtype = float)
        B = np.zeros(shape = (A.shape[0] ,1) ,dtype = float)
        
        k = 0
        for i in range(0,z.shape[0]):
            for j in range(0,z.shape[1]):
                wij = self.compute_weight(z[i][j])
                A[k][z[i][j]] = wij
                A[k][n+i+1] = -wij
                B[k][0] = wij*exp[j]
                k=k+1
                
        A[k][128] =1
        k=k+1
        

        for i in range(0,n-1):
            A[k][i] = l*self.compute_weight(i+1)
            A[k][i+1] = -2*l*self.compute_weight(i+1)
            A[k][i+2] = l*self.compute_weight(i+1)
            k=k+1
            
        x, residuals, rank, s = np.linalg.lstsq(A, B , rcond=None)
        
        g = x[:256]
        le=x[256:]
        
        return g[:,0]
    
    def HDR_Sloution(self ,z , Image , exp , l , output_dir ):
        hdr = np.ndarray(shape = (Image[0].shape), dtype = float)
        rad = np.ndarray(shape = (Image[0].shape), dtype = float)
        for i in range(0,3):# 通道個別計算        
            Gcurve =  self.generate_curve(z[i], exp, l) 
            rad[:,:,i] = self.compute_radiance(Image[:,:,:,i], exp, Gcurve)
            hdr[:,:,i]  = cv2.normalize(rad[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            plt.figure(i,figsize = (10,10) )
            if i == 0:
                plt.title("Red")
                plt.plot(Gcurve,range(256) , i , color='r')
                plt.ylabel('Z value ')
                plt.xlabel('X Log ')
                plt.show()
                plt.savefig(output_dir+'Red_Curve.jpg')
            elif i == 1 :
                plt.title("Green")
                plt.plot(Gcurve,range(256) , i , color='g')
                plt.ylabel('Z value ')
                plt.xlabel('X Log ')
                plt.show()
                plt.savefig(output_dir+'Green_Curve.jpg')
            else:
                plt.title("Blue")
                plt.plot(Gcurve,range(256) , i , color='b')
                plt.ylabel('Z value ')
                plt.xlabel('X Log ')
                plt.show()
                plt.savefig(output_dir+'Blue_Curve.jpg')

            
        return rad, hdr
    
    def compute_radiance(self ,Image , exp , Gcurve):
        
        rad = np.zeros(shape=(Image.shape[1] , Image.shape[2]), dtype=float)
        imgNums = Image.shape[0]

        for i in range(0, Image.shape[1]):
            for j in range(0, Image.shape[2]):
                g = np.ndarray(shape = imgNums, dtype=float)
                w = np.ndarray(shape = imgNums, dtype=float)
                for k in range(0, imgNums):
                    g[k] = Gcurve[Image[k][i][j]]
                    w = self.compute_weight(Image[k][i][j])
    
                totalW = np.sum(w)
                if totalW > 0:
                    rad[i][j] = np.sum(w * (g - exp) / totalW) # g-曝光時間/w總和
                else:
                    rad[i][j] = np.sum(w * (g - exp))
        return rad
    
    def computeRadianceMap(self,output_dir,img):
    
     
    
        from matplotlib.pylab import cm
        colorize = cm.jet
        cmap =( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255)
        # cmap = colorize(cmap)
        cmap =  cv2.applyColorMap(cmap.astype('uint8'), cv2.COLORMAP_JET)
        cv2.imwrite(output_dir+'cmap.jpg', np.uint8(cmap*255.))
        
        cv2.imshow("cmap" , cmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return cmap
    
    def SaveHDRformat(self, output_dir,hdr):
        cv2.imwrite(output_dir+"03.hdr",hdr)
        