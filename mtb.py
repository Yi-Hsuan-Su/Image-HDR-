import cv2
import numpy as np



class ImageAlignment():
    
    def gray(self, img):
        temp = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        return temp

    def imageShrink2(self, img):
        retImg = cv2.resize(img, (img.shape[0], img.shape[1]))
        return retImg

    def computeBitmaps(self, img):
        med = int(np.median(img))
        thresBitmap = np.array([[True if yi > med else False for yi in xi] for xi in img], dtype='bool')

        x, y = img.shape
        exclusionBitmap = np.full((x, y), True, dtype='bool')
        for i in range(x):
            for j in range(y):
                if abs(img[i][j] - med) < 5:
                    exclusionBitmap[i][j] = False
        return (thresBitmap, exclusionBitmap)

    def bitmapShift(self, bitmap, x, y):
        shifted = np.full(bitmap.shape, False, dtype='bool')
        if x > 0:
            shifted[x:] = bitmap[:-x]
        elif x < 0:
            shifted[:x] = bitmap[-x:]
        else:
            shifted = bitmap
        if y > 0:
            shifted = [np.append([False] * y, row[:-y]) for row in shifted]
        elif y < 0:
            shifted = [np.append(row[-y:], [False] * -y) for row in shifted]
        return shifted

    def getExpShift(self, img0, img1, shiftBits):
        if shiftBits > 0:
            sml_Img0 = self.imageShrink2(img0)
            sml_Img1 = self.imageShrink2(img1)
            curShiftBits = self.getExpShift(sml_Img0, sml_Img1, shiftBits-1)
            curShiftBits[0] *= 2
            curShiftBits[1] *= 2
        else:
            curShiftBits = [0, 0]
        tb0, eb0 = self.computeBitmaps(img0)
        tb1, eb1 = self.computeBitmaps(img1)
        minErr = img0.shape[0] * img0.shape[1]
        for i in range(-1, 2):
            for j in range(-1, 2):
                xs = curShiftBits[0] + i
                ys = curShiftBits[1] + j
                shifted_tb1 = self.bitmapShift(tb1, xs, ys)
                shifted_eb1 = self.bitmapShift(eb1, xs, ys)
                diff_b = np.logical_xor(tb0, shifted_tb1)
                diff_b = np.logical_and(diff_b, eb0)
                diff_b = np.logical_and(diff_b, shifted_eb1)
                err = np.sum(diff_b)
                if err < minErr:
                    ret = [xs, ys]
                    minErr = err
        return ret

    def imgShift(self, img, x, y):
        shifted = np.full(img.shape, 0, dtype='uint8')
        if x > 0:
            shifted[x:] = img[:-x]
        elif x < 0:
            shifted[:x] = img[-x:]
        else:
            shifted = img
        if y > 0:
            shifted = [np.concatenate([[[0, 0, 0]] * y, row[:-y]]) for row in shifted]
        elif y < 0:
            shifted = [np.concatenate([row[-y:], [[0, 0, 0]] * -y]) for row in shifted]
        return shifted

    def align(self, img0, img1, shiftBits):
        g0 = self.gray(img0)
        g1 = self.gray(img1)
        return self.getExpShift(g0, g1, shiftBits)

    def process(self, imgs_src, shiftBits):
        ret = [imgs_src[0]]
        if len(imgs_src) < 2:
            return ret
        else:
            for i in range(1, len(imgs_src)):
                x, y = self.align(imgs_src[0], imgs_src[1], shiftBits)
                ret.append(self.imgShift(imgs_src[i], x, y))
            return ret