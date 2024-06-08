import cv2
import math
import numpy as np
import random as rnd
import matplotlib.pyplot as plt


from vars.model import load_sam


class util ():
    def __init__(self, path) -> None:
        self.mask_generator = load_sam()
        self.path = path
        self.image = self.image_load()

    def image_load(self):
        # Path to your image file

        # Load and display the image
        image = cv2.imread(self.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (0, 0), fx=1000 /
            image.shape[1], fy=800/image.shape[0])
        self.image = image

    def image_visulazation(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image)
        plt.axis('off')
        plt.show()

    def segment_img(self):
        self.masks = self.mask_generator.generate(self.image)
        print(len(self.masks))
        print(self.masks[0].keys())
        for i in range(30):
            plt.subplot(5, 6, i+1)
            plt.imshow(self.masks[i+1]["segmentation"])

    def segment_selection(self, img_no):
        self.img_no = img_no
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 2, 1)
        plt.imshow(self.image)

        plt.subplot(1, 2, 2)
        plt.imshow(self.masks[img_no]["segmentation"])
        plt.axis('off')
        plt.show()
        self.image.shape
        print(self.masks[img_no]["bbox"])

    def seperate_segment(self):
        # for all masks
        mask_list = []
        seg_obj_list = []
        for i in range(len(self.masks)):
            mask_list.append(self.masks[i]["segmentation"])
        for mask in mask_list:
            pixel_list = []
            for i in range(len(self.image)):
                for j in range(len(self.image[i])):
                    if (not (mask[i][j])):
                        pixel_list.append([255, 255, 255])
                    else:
                        pixel_list.append(self.image[i][j])

            segmented_obj = np.array(pixel_list).reshape(
                len(self.image), len(self.image[0]), 3)
            seg_obj_list.append(segmented_obj)
        plt.imshow(seg_obj_list[self.img_no])

    def msng_pxl_flr(self):
        car_mask = self.masks[self.img_no]["segmentation"]
        pixel_list = []
        for i in range(len(self.image)):
            for j in range(len(self.image[i])):
                if (car_mask[i][j] == True):
                    pixel_list.append([255, 255, 255])
                else:
                    pixel_list.append(self.image[i][j])

        self.segmented_obj = np.array(pixel_list).reshape(
            len(self.image), len(self.image[0]), 3)
        plt.imshow(self.segmented_obj)

    def get_neighboring_pixel(self, img, x, y):
        x_rand, y_rand = 0, 0

        max_num_tries = 30
        max_tries_per_neighbourhood = 5
        neighbourhood_size_increment = 30
        current_window_size = 30
        total_tries = 3
        for _ in range(math.ceil(max_num_tries/max_tries_per_neighbourhood)):
            for _ in range(max_tries_per_neighbourhood):
                min_x = max(0, x-current_window_size)
                max_x = min(800, x+current_window_size)
                min_y = max(0, y-current_window_size)
                max_y = min(1000, y+current_window_size)
                x_rand = rnd.randint(min_x, max_x-1)
                y_rand = rnd.randint(min_y, max_y-1)
                total_tries += 1
                if not (img[x_rand][y_rand][0] == 0 and img[x_rand][y_rand][1] == 0 and img[x_rand][y_rand][2] == 0):
                    return x_rand, y_rand
                current_window_size += neighbourhood_size_increment

        return x_rand, y_rand

    def fill_swath_with_neighboring_pixel(self):
        img_with_neighbor_filled = self.segmented_obj.copy()
        (x_swath, y_swath, z_swath) = np.where(
            self.segmented_obj == [255, 255, 255])
        # print((x_swath, y_swath, z_swath))

        for i in range(len(x_swath)):
            x_rand, y_rand = self.get_neighboring_pixel(
                self.segmented_obj, x_swath[i], y_swath[i])
            img_with_neighbor_filled[x_swath[i]
                                     ][y_swath[i]] = self.segmented_obj[x_rand][y_rand]
        return img_with_neighbor_filled

    def ff11(self):
        for i in range(9):
            segmented_obj = self.fill_swath_with_neighboring_pixel()

        plt.imshow(segmented_obj)
