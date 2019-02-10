import os
import numpy as np
import cv2
import time
import io
import multiprocessing
import threading
from PIL import Image, ImageFile
from scipy.signal import savgol_filter

MAIN_KEY = 'main'
MASK_KEY = 'mask'
LIGHT_PIXEL_PERCENTAGE = 90
ImageFile.LOAD_TRUNCATED_IMAGES = True
first = 'image/1'
output = 'image/result'


def connect_mask_and_main(folder):
    mask_dict = dict()
    main_dict = dict()
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if MAIN_KEY in name:
            key_index = name.index(MAIN_KEY)
            main_dict[name[:key_index - 1]] = path
        elif MASK_KEY in name:
            key_index = name.index(MASK_KEY)
            mask_dict[name[:key_index - 1]] = path

    result_list = []
    for key in mask_dict:
        if key in main_dict.keys():
            result_list.append((mask_dict[key], main_dict[key]))

    return result_list


def is_light(value):
    return value >= 470


def mostly_white_line(value, amount, standard):
    return (float(value) / amount) * 100 >= standard


def get_base_crop_area(mask_pixels_matrix, matrix_width, matrix_height, standard):
    dark_area = get_dark_crop_area(mask_pixels_matrix, matrix_width, matrix_height, standard)
    return get_light_crop_area(mask_pixels_matrix, dark_area[2], dark_area[3], dark_area[0], dark_area[1], 95)


def get_dark_crop_area(mask_pixels_matrix, matrix_width, matrix_height, standard):
    return get_base_dark2_crop_area(mask_pixels_matrix, matrix_width, matrix_height, standard,
                                    width_step=matrix_width // 100, height_step=matrix_height // 100)


def get_light_crop_area(mask_pixels_matrix, matrix_width, matrix_height, start_x, start_y, standard):
    return get_base_light_crop_area(mask_pixels_matrix, matrix_width, matrix_height, standard, start_x, start_y,
                                    width_step=matrix_width // 100, height_step=matrix_height // 100)


def get_base_dark_crop_area(mask_pixels_matrix, matrix_width, matrix_height, standard, start_x=0, start_y=0,
                            width_step=1, height_step=1):
    left_point = -1
    top_point = -1

    for a in range(start_x, matrix_width // 2, width_step):
        left_counter = 0
        right_counter = 0
        for b in range(start_y, matrix_height):
            if is_light(sum(mask_pixels_matrix[a, b])):
                left_counter += 1
            if is_light(sum(mask_pixels_matrix[matrix_width + start_x - a - 1, b])):
                right_counter += 1
        if mostly_white_line(left_counter, matrix_height - start_y, standard) \
                and mostly_white_line(right_counter, matrix_height - start_y, standard):
            left_point = a
            if left_point > start_x:
                start_x = left_point
                matrix_width -= start_x
            break

    for b in range(start_y, matrix_height // 2, height_step):
        top_counter = 0
        bot_counter = 0
        for a in range(start_x, matrix_width):
            if is_light(sum(mask_pixels_matrix[a, b])):
                top_counter += 1
            if is_light(sum(mask_pixels_matrix[a, matrix_height + start_y - b - 1])):
                bot_counter += 1
        if mostly_white_line(top_counter, matrix_width - left_point, standard) \
                and mostly_white_line(bot_counter, matrix_width - left_point, standard):
            top_point = b
            if top_point > start_y:
                start_y = top_point
                matrix_height -= start_y
            break

    if top_point == -1 or left_point == -1:
        return start_x, start_y, matrix_width, matrix_height
    elif standard >= 99:
        return left_point, top_point, matrix_width, matrix_height
    else:
        next_standard = standard + int(standard * 0.1)
        standard = next_standard if next_standard < 99 else 99
        width_step = width_step // 2 if width_step // 2 > 1 else 1
        height_step = height_step // 2 if height_step // 2 > 1 else 1
        return get_base_dark_crop_area(mask_pixels_matrix, matrix_width, matrix_height, standard, start_x,
                                       start_y, width_step, height_step)


def get_base_light_crop_area(mask_pixels_matrix, matrix_width, matrix_height, standard, start_x=0, start_y=0,
                             width_step=1, height_step=1):
    l_dark = -1
    r_dark = -1
    t_dark = -1
    b_dark = -1
    for a in range(start_x, matrix_width, width_step):
        if l_dark != -1 and r_dark != -1:
            break

        left_counter = 0
        right_counter = 0
        for b in range(start_y, matrix_height):
            if l_dark == -1 and is_light(sum(mask_pixels_matrix[a, b])):
                left_counter += 1
            if r_dark == -1 and is_light(sum(mask_pixels_matrix[matrix_width + start_x - a - 1, b])):
                right_counter += 1
        if l_dark == -1 and not mostly_white_line(left_counter, matrix_height - start_y, standard):
            l_dark = a - width_step if a - width_step > start_x else start_x
        if r_dark == -1 and not mostly_white_line(right_counter, matrix_height - start_y, standard):
            next_right = matrix_width - a + width_step + start_x
            r_dark = next_right if next_right < matrix_width else matrix_width

    for b in range(start_y, matrix_height, height_step):
        if t_dark != -1 and b_dark != -1:
            break

        top_counter = 0
        bot_counter = 0
        for a in range(start_x, matrix_width):
            if t_dark == -1 and is_light(sum(mask_pixels_matrix[a, b])):
                top_counter += 1
            if b_dark == -1 and is_light(sum(mask_pixels_matrix[a, matrix_height + start_y - b - 1])):
                bot_counter += 1
        if t_dark == -1 and not mostly_white_line(top_counter, matrix_width - start_x, standard):
            t_dark = b - height_step if b - height_step > start_y else start_y
        if b_dark == -1 and not mostly_white_line(bot_counter, matrix_width - start_x, standard):
            next_b = matrix_height - b + height_step + start_y
            b_dark = next_b if next_b < matrix_height else matrix_height

    return l_dark, t_dark, r_dark - 1, b_dark - 1


def get_base_dark2_crop_area(mask_pixels_matrix, matrix_width, matrix_height, standard, start_x=0, start_y=0,
                             width_step=1, height_step=1):
    l_dark = -1
    r_dark = -1
    t_dark = -1
    b_dark = -1
    for a in range(start_x, matrix_width, width_step):
        if l_dark != -1 and r_dark != -1:
            break

        left_counter = 0
        right_counter = 0
        for b in range(start_y, matrix_height):
            if l_dark == -1 and is_light(sum(mask_pixels_matrix[a, b])):
                left_counter += 1
            if r_dark == -1 and is_light(sum(mask_pixels_matrix[matrix_width + start_x - a - 1, b])):
                right_counter += 1
        if l_dark == -1 and mostly_white_line(left_counter, matrix_height - start_y, standard):
            l_dark = a - width_step if a - width_step > start_x else start_x
        if r_dark == -1 and mostly_white_line(right_counter, matrix_height - start_y, standard):
            next_right = matrix_width - a + width_step + start_x
            r_dark = next_right if next_right < matrix_width else matrix_width

    for b1 in range(start_y, matrix_height, height_step):
        if t_dark != -1 and b_dark != -1:
            break

        top_counter = 0
        bot_counter = 0
        for a1 in range(l_dark, r_dark):
            if t_dark == -1 and is_light(sum(mask_pixels_matrix[a1, b1])):
                top_counter += 1
            if b_dark == -1 and is_light(sum(mask_pixels_matrix[a1, matrix_height + start_y - b1 - 1])):
                bot_counter += 1
        if t_dark == -1 and mostly_white_line(top_counter, r_dark - l_dark, standard):
            t_dark = b1 - height_step if b1 - height_step > start_y else start_y
        if b_dark == -1 and mostly_white_line(bot_counter, r_dark - l_dark, standard):
            next_b = matrix_height - b1 + height_step + start_y
            b_dark = next_b if next_b < matrix_height else matrix_height

    # return l_dark, t_dark, r_dark - 1, b_dark - 1
    if l_dark == -1 or t_dark == -1 or r_dark == -1 or b_dark == -1:
        return start_x, start_y, matrix_width, matrix_height
    elif standard >= 99:
        return l_dark, t_dark, r_dark - 1, b_dark - 1
    else:
        next_standard = standard + int(standard * 0.1)
        standard = next_standard if next_standard < 99 else 99
        width_step = width_step // 2 if width_step // 2 > 1 else 1
        height_step = height_step // 2 if height_step // 2 > 1 else 1
        return get_base_dark2_crop_area(mask_pixels_matrix, r_dark, b_dark, standard, l_dark,
                                        t_dark, width_step, height_step)


def getSobel(channel):
    sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
    sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
    sobel = np.hypot(sobelx, sobely)

    return sobel;


def findSignificantContours(img, sobel_8u):
    contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = sobel_8u.size * 5 / 100  # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:

        contour = contours[tupl[0]];

        epsilon = 0.10 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 3, True)
        contour = approx

        # window_size = int(
        #     round(min(img.shape[0], img.shape[1]) * 0.05))  # Consider each window to be 5% of image dimensions
        # x = savgol_filter(contour[:, 0, 0], window_size * 2 + 1, 3, mode='wrap')
        # y = savgol_filter(contour[:, 0, 1], window_size * 2 + 1, 3, mode='wrap')
        #
        # approx = np.empty((x.size, 1, 2))
        # approx[:, 0, 0] = x
        # approx[:, 0, 1] = y
        # approx = approx.astype(int)
        # contour = approx

        area = cv2.contourArea(contour)
        if area > tooSmall:
            cv2.drawContours(img, [contour], 0, (255, 255, 255), 2, cv2.LINE_AA, maxLevel=1)
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant];


def segment(path_main, path_mask, need_to_rotate, name):
    mask_buf = np.fromstring(path_mask.getvalue(), np.uint8)
    main_buf = np.fromstring(path_main.getvalue(), np.uint8)

    img_mask = cv2.imdecode(mask_buf, flags=cv2.IMREAD_UNCHANGED)
    img_main = cv2.imdecode(main_buf, flags=cv2.IMREAD_UNCHANGED)

    blurred = cv2.GaussianBlur(img_mask, (5, 5), 0)  # Remove noise

    # Edge operator
    sobel = np.max(np.array([getSobel(blurred[:, :, 0]), getSobel(blurred[:, :, 1]), getSobel(blurred[:, :, 2])]),
                   axis=0)

    # Noise reduction trick, from http://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l182
    mean = np.mean(sobel)

    # Zero any values less than mean. This reduces a lot of noise.
    sobel[sobel <= mean] = 0
    sobel[sobel > 255] = 255

    # cv2.imwrite('image/result/edge.png', sobel);

    sobel_8u = np.asarray(sobel, np.uint8)

    # Find contours
    significant = findSignificantContours(img_main, sobel_8u)

    # Mask
    mask = sobel.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)

    # Finally remove the background
    img_main[mask] = 255;

    if need_to_rotate:
        img_main = rotate_image(img_main, 270)

    fname = name.split('/')[-1]

    print (output + fname)
    cv2.imwrite(output + '/' + fname, img_main);


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def crop_image(start_time, mask, main):
    print ('start ' + main)
    mask_img = Image.open(mask)  # open mask photo
    main_img = Image.open(main)
    width, height = mask_img.size
    need_to_rotate = width > height
    light_base_crop_area = get_base_crop_area(mask_img.load(), width, height, 50)
    mask_img = mask_img.crop(light_base_crop_area)
    main_img = main_img.crop(light_base_crop_area)
    maskBytes = io.BytesIO()
    mainBytes = io.BytesIO()
    mask_img.save(maskBytes, format='jpeg')
    main_img.save(mainBytes, format='jpeg')
    mask_img.close()
    main_img.close()
    segment(path_mask=maskBytes, path_main=mainBytes, need_to_rotate=need_to_rotate, name=main)
    print ('finished in: ' + str(time.time() - start_time))


def execute(folder):
    start_time = time.time()
    for pair in connect_mask_and_main(folder):
        mask, main = pair
        p = multiprocessing.Process(target=crop_image, args=(start_time, mask, main))
        p.start()
        # crop_image(mask, main)
    # print('cropped all images in: ' + str(time.time() - start_time))


execute(first)
