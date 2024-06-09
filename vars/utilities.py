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

    def fill_swath_with_neighboring_pixel(img, num_iterations=10):
        img_with_neighbor_filled = img.copy()
        white_mask = np.all(img == [255, 255, 255], axis=-1)

        for _ in range(num_iterations):
            white_indices = np.argwhere(white_mask)
            np.random.shuffle(white_indices)

            for x, y in white_indices:
                neighboring_pixel = get_neighboring_pixel(
                    img_with_neighbor_filled, x, y)
                img_with_neighbor_filled[x, y] = neighboring_pixel

            white_mask = np.all(img_with_neighbor_filled ==
                                [255, 255, 255], axis=-1)

            if not np.any(white_mask):
                break

        return img_with_neighbor_filled

    def get_neighboring_pixel(img, x, y, window_size=30):
        min_x = max(0, x - window_size)
        max_x = min(img.shape[0], x + window_size + 1)
        min_y = max(0, y - window_size)
        max_y = min(img.shape[1], y + window_size + 1)

        window = img[min_x:max_x, min_y:max_y]
        non_white_pixels = window[np.where(
            np.any(window != [255, 255, 255], axis=-1))]

        if len(non_white_pixels) > 0:
            random_pixel = non_white_pixels[np.random.choice(
                len(non_white_pixels))]
            return random_pixel
        else:
            return img[x, y]

    def auto_mask():

        # Define the transformation
        transform_resize = transforms.Compose([
            transforms.Resize((32, 32))
        ])

        # Select the car mask automatically

        car_masks = []
        model_inputs = []
        for n, mask in enumerate(masks):
            image_tensor = transform(image)
            bb_x = mask['bbox'][0]
            bb_y = mask['bbox'][1]
            bb_w = mask['bbox'][2]
            bb_h = mask['bbox'][3]
            image_crop = transform_resize(
                image_tensor[:, bb_y: bb_y + bb_h, bb_x: bb_x + bb_w])

            output = car_detect_network(image_crop)
            _, predicted = torch.max(output.data, 1)
            if predicted.item() == 1:  # Automobile class
                car_masks.append(mask['segmentation'])
                print(f"found car at position {n}")

        # Plot the car masks on the source image
        if len(car_masks) > 0:
            masked_image = image.copy()
            for car_mask in car_masks:
                # Set the mask pixels to white
                masked_image[car_mask] = [255, 255, 255]

            plt.figure(figsize=(10, 10))
            plt.imshow(masked_image)
            plt.axis('off')
            plt.show()
        else:
            print("No cars detected in the image.")
