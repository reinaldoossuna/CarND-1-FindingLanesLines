import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from pathlib import Path
from moviepy.editor import VideoFileClip
from line_class import Line

def test_functions(path,foo=None,cmap=None):
    """
    Function to test a function (foo) with a folder of imgs (path)
    All images in the folder will be printed.

    Args:
        path: path to images folders
        foo: function
    Return:
        None

    """

    p = Path(path)
    imgs = list(p.glob("*.jpg"))
    fig, axs = plt.subplots(2,len(imgs)//2,figsize=(30,10))
    axs = axs.flatten()
    for i,img in enumerate(imgs):
        image = mpimg.imread(img)
        if foo:
            image = foo(image)
        if len(image.shape) < 3:
            cmap = "gray"
        axs[i].imshow(image,cmap=cmap)

def color_selection(img, threshold):
    """
    Return a mask from the image from pixel between the threshold
    Args:
        img: RGB image. Use the mpimg.imread() 

        threshold: Tuple with the low and high threshold from a HLS image

         np.array([0, 200, 0]), np.array([255, 255,70]))
    return:
        Black and white image. White pixel are between the threshold

    """
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    mask = cv2.inRange(img_hls, *threshold)
    return mask

def yellow_white_selection(img):
    """
    Return a mask from the white and yellow_threshold

    white_threshold =  [0, 200, 0] [255, 255,70]
    
    yellow_threshold = [0, 80, 150] [80, 180, 255]

    If you want to use another values, Use the color_selection function!

    Args:
        img: RGB image. Use the mpimg.imread() 

    return:
        Black and white image. White pixel are between the threshold

    """
    white_threshold =  (np.array([0, 200, 0]), np.array([255, 255,70]))
    white_mask = color_selection(img, white_threshold)
    
    yellow_threshold =  (np.array([0, 80, 150]), np.array([80, 180, 255]))
    yellow_mask = color_selection(img, yellow_threshold)
    
    color_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_or(img, img, mask=color_mask)

def region_of_interest(img, vertices=None):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    if not vertices:
        imshape = img.shape
        vertices = np.array([[(imshape[1]/17,imshape[0]),
                      (4*imshape[1]/9, 9*imshape[0]/15), 
                      (5*imshape[1]/9, 9*imshape[0]/15), 
                
                      (16*imshape[1]/17,imshape[0])]], dtype=np.int32)
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def byte_image(img):
    """
    Return a bit img. (Image with 0 or 255 pixels )
    
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray[gray > 0] = 255
    return gray

def fit_line(side_img):
    """

    Fit a line in all pixel above 0.

    Args:
        side_img: 
            Half image. It's expected to the right and left line be separated in each "half image"
    return:
        f: function of the fit line
    
    """

    points = np.where(side_img > 0)
    f = np.poly1d(np.polyfit(*points,1))
    return f

def points(side_img,x = 0):
    """
    Return two extremes points from the line
    
    """
    # Same values used in the region_of_interest()

    y2 = int(9*side_img.shape[0]/15)
    y1 = int(side_img.shape[0])

    f = fit_line(side_img)

    return [int(f(y1))+x,y1,int(f(y2))+x,y2]


def weighted_img(line_img, initial_img, α=0.8, β=1., γ=0.):
    """
    `line_img` is an image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, line_img, β, γ)


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    for line in lines:
        x1,y1,x2,y2 =  line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness,)

def test_video(path, foo, path_out):
    clip1 = VideoFileClip(path)
    
    clip = clip1.fl_image(foo) #NOTE: this function expects color images!!
    clip.write_videofile(path_out, audio=False)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def gaussian_blur(img, kernel_size=5):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def billateral_blur(img, d=9,
                         sigmaColor=120,
                         sigmaSpace=120):
    """Applies a Gaussian Noise kernel"""
    return cv2.bilateralFilter(img,d,sigmaColor,sigmaSpace)
  

def canny_edge(img, low_threshold=50,high_threshold=150):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def hough_lines(img, 
                rho=2,
                theta=np.pi/180,
                threshold=20,
                min_line_length=40,
                max_line_gap=15):
    """
    `img` should be the output of a Canny transform.
        
    Returns an list with hough lines .
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)

    return lines

def fit_lines(lines):
    """
    Args:
        lines: List of hough Lines

    return:
        The right Line and the left line
    
    """
    right_line = None
    left_line = None
    for line in lines:
        for x1,y1,x2,y2 in line:

            l1 = Line([x1,x2],[y1,y2])
            if l1.is_vertical():
                if l1.is_left_line():
                    if not left_line:
                        left_line = l1
                    else:
                        left_line = left_line.fit(l1)
                else:
                    if not right_line:
                        right_line = l1
                    else:
                        right_line = right_line.fit(l1)
            else:
                pass
    return right_line, left_line
