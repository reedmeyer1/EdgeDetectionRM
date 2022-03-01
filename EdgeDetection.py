import cv2
from matplotlib import pyplot as plt

def Main():
    source = cv2.imread('./envelope.jpg', 1)
    sourceFinal = cv2.imread('./envelope.jpg', 1)
    sourceGBR = cv2.imread('./envelope.jpg', 1)
    sourceRGB = cv2.cvtColor(sourceGBR, cv2.COLOR_BGR2RGB)
    
    sourceEdge = EdgeDetect(source)
    sourceMap = Overlay(sourceRGB, sourceEdge, sourceFinal)
    
    Plot(sourceRGB, sourceFinal, sourceEdge)
    
    plt.show()

def EdgeDetect(source):
    sourceBlur = cv2.blur(source, (20,20))
    sourceCanny = cv2.Canny(sourceBlur, 5, 10)
    
    return sourceCanny

def Overlay(source, sourceEdge, sourceFinal):
    greenMap = cv2.imread('./green.jpg', 1)
    sourceBlurredEdge = cv2.blur(sourceEdge, (10,10))
    sourceSub = cv2.subtract(greenMap, source, sourceFinal, sourceBlurredEdge, 0)
    sourceFinal = cv2.blur(sourceSub, (10,10))
    return sourceFinal
    
def Plot(source, sourceMap, sourceEdge):
    plt.subplot(131),plt.imshow(source)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(sourceMap)
    plt.title('Overlay'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(sourceEdge)
    plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])

Main()
