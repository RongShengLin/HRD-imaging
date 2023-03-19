import cv2 
import numpy as np

# call this function
# input : list of images
# output : list of aligned images
def alignment(images, level = 8):
    aligned_images = [images[0]]
    binary_images = get_binary_images(images)
    for i in range(1, len(binary_images)):
        dx, dy = align_two(binary_images[i-1], binary_images[i], level)
        #print("dx dy are ", dx, dy)
        binary_images[i] = shift_image(binary_images[i], dx, dy)
        aligned_images.append(shift_image(images[i], dx, dy))
    return aligned_images

def get_grayscale_image(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixel_list = []
    median = 0
    for i in range(len(new_image)):
        pixel_list += list(new_image[i])
    # find median as threshold
    list.sort(pixel_list)
    l = len(pixel_list)
    if l%2 == 0:
        median = (int(pixel_list[l//2])+int(pixel_list[l//2+1]))/2
    else :
        median = pixel_list[l//2]
    _, grayscale_image = cv2.threshold(new_image, median, 255, 0)

    # compute Exclusion bitmap, ignore noise near the median(near mdeian -> 0, else -> 255) 
    mask = cv2.inRange(image, median-10, median+10)
    # replace 255 with 1
    mask = transform_to_binary(~mask)
    grayscale_image = transform_to_binary(grayscale_image)
    return mask, grayscale_image

def transform_to_binary(image):
    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j] = 1 if image[i][j] == 255 else 0
    return image 
    
def get_binary_images(images):
    binary_images = []
    for i in range(len(images)):
        mask, grayscale_image = get_grayscale_image(images[i])
        binary_image = BitAND(grayscale_image, mask)
        binary_images.append(binary_image)
    return binary_images

def shift_image(image, dx, dy):
    shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    col, row = np.shape(image)[0], np.shape(image)[1]
    new_image = cv2.warpAffine(image, shift_matrix, (row, col))
    return new_image

def shrink_images(image):
    col, row = np.shape(image)[0], np.shape(image)[1]
    new_image = cv2.resize(image, (int(row/2), int(col/2)), interpolation=cv2.INTER_LINEAR)
    return new_image

def BitXOR(image1, image2):
    bit_image = image1.copy() 
    for i in range(len(image1)):
        for j in range(len(image1[i])):
            bit_image[i][j] = image1[i][j]^image2[i][j]
    return bit_image

def BitAND(image1, image2):
    bit_image = image1.copy() 
    for i in range(len(image1)):
        for j in range(len(image1[i])):
            bit_image[i][j] = image1[i][j]&image2[i][j]
    return bit_image

def count_diff(image1, image2):
    diff_image = BitXOR(image1, image2)
    sum = 0
    for i in range(len(diff_image)):
        for j in range(len(diff_image[i])):
            sum += diff_image[i][j]
    return sum

def align_two(image1, image2, level):
    dx = dy = 0
    if level > 0 :
        next_image1 = shrink_images(image1)
        next_image2 = shrink_images(image2)
        dx, dy = align_two(next_image1, next_image2, level-1)
        dx *= 2
        dy *= 2

    min_diff = 10**8
    dir = [0, 0]
    for i in [1, -1, 0]:
        for j in [1, -1, 0]:
            shifted_image2 = shift_image(image2, dx+i, dy+j)
            diff = count_diff(image1, shifted_image2)
            if diff < min_diff :
                min_diff = diff
                dir[0] = i
                dir[1] = j
                #print(min_diff, dx + dir[0], dy + dir[1])
    #print(min_diff, dx + dir[0], dy + dir[1])
    #print("------------")
    return dx + dir[0], dy + dir[1] 

# test
def test():
    images = []
    images.append(cv2.imread("./Memorial_SourceImages/memorial0065.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0069.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0070.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0071.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0072.png"))
    images.append(cv2.imread("./Memorial_SourceImages/memorial0073.png"))
    aligned_images = alignment(images)
    for i in [0, 3, 4, 5]:
        cv2.imshow("aligned_image"+str(i), aligned_images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
test()