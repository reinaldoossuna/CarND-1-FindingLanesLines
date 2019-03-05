import numpy as np
import cv2

class Line():

    """ Line Class """
    
    def __init__(self,x,y,m = None, b= None , w = None):
        """ Args:
                x: List with the x's values from the line 
                y: List with the y's values from the line
                m: slope from the line
                b: b values from the equation
                w: weights to fit lines. I used the size of line
        """
                
        self.X = np.array(x)
        self.Y = np.array(y)

        if w is None:
            self.calculate_weight(x,y)
        else:
            self.w = w
            
        if m is None or b is None:
            self.calculate_eq()
        else:
            self.m = m
            self.b = b
        
        
    def calculate_eq(self):
        """ 
        Calculate the equation from the line
            y = m X * b
            
        """
        self.m, self.b = np.polyfit(self.X,self.Y,1)
        
    def calculate_weight(self,x,y):
        """
        The weight will be used in the fit(). 
        Larger lines have a higher weight
        """
        x1,x2 = x
        y1,y2 = y
        size = np.sqrt(np.power(x2- x1,2)+np.power(y2-y1,2))
        self.w = np.array([size] * 2)
        
    def is_vertical(self,m_threshold=0.3):
        """ Return True if the slope is greater (or lower then) """
        
        return (self.m > m_threshold or self.m < -m_threshold)

    def is_left_line(self):
        """ Return True if the line is The left line """
        return self.m > 0
    
    def fit(self, other):

        """ 
        fit this line with other Line.
        Both lines need to be vertical, and from the same side
        
        Return:
            Line class with a fitting line from the both lines
        """
        if all([self.is_vertical(), other.is_vertical()]):
            if self.is_left_line() is other.is_left_line():
                
                X_ = np.append(self.X, other.X)
                Y_ = np.append(self.Y, other.Y)
                W = np.append(self.w, other.w)
                m, b = np.polyfit(X_,Y_,1,w = W)
                return Line(X_,Y_,m,b,W)

            
    def draw(self,img, color=[255, 0, 0], thickness=8):

        """

        This function draws `lines` with `color` and `thickness`.    
        Lines are drawn on the image inplace (mutates the image).
        """
        imshape = img.shape

        # Values from the region_of_interest
        min_y = int(9*imshape[0]/15)
        max_y = imshape[0]

        # How i have the y's a will calculate the inverse function herek
        gy = np.poly1d([1/self.m, -self.b/self.m])
        cv2.line(img, (int(gy(min_y)),min_y), (int(gy(max_y)),max_y), color, thickness)
