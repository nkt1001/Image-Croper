import os
import numpy as np
import cv2
import time
# import multiprocessing
# from scipy.signal import savgol_filter

MAIN_KEY = 'main'
MASK_KEY = 'mask'
LIGHT_MEAN_STANDARD = 470
first = 'image/debug'
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


def is_line_light(array, standard):
    length = len(array)
    number = len(np.where(np.sum(array, axis=1) > LIGHT_MEAN_STANDARD)[0])
    return float(number) / length * 100 >= standard


def get_crop_area_dark(img, standard, x_start, x_end, y_start, y_end,  width_step=1, height_step=1):
    l_dark = -1
    r_dark = -1
    t_dark = -1
    b_dark = -1
    for a in range(x_start, x_end, width_step):
        if l_dark != -1 and r_dark != -1:
            break

        if l_dark == -1 and is_line_light(img[y_start:y_end, a], standard):
            l_dark = a
        if r_dark == -1 and is_line_light(img[y_start:y_end, x_end + x_start - a - 1], standard):
            r_dark = x_end - a + x_start

    for b in range(y_start, y_end, height_step):
        if t_dark != -1 and b_dark != -1:
            break

        if t_dark == -1 and is_line_light(img[b, l_dark:r_dark], standard):
            t_dark = b
        if b_dark == -1 and is_line_light(img[y_end - b + y_start - 1, l_dark:r_dark], standard):
            b_dark = y_end - b + y_start

    if l_dark == -1 or t_dark == -1 or r_dark == -1 or b_dark == -1:
        return y_start, y_end, x_start, x_end
    elif standard >= 99:
        return t_dark, b_dark, l_dark, r_dark
    else:
        next_standard = standard + int(standard * 0.1)
        standard = next_standard if next_standard < 99 else 99
        width_step = width_step // 2 if width_step // 2 > 1 else 1
        height_step = height_step // 2 if height_step // 2 > 1 else 1
        return get_crop_area_dark(img, standard, l_dark, r_dark, t_dark, b_dark,
                                  width_step, height_step)


def get_crop_area_light(img, standard, x_start, x_end, y_start, y_end,  width_step=1, height_step=1):
    l_dark = -1
    r_dark = -1
    t_dark = -1
    b_dark = -1
    for a in range(x_start, x_end, width_step):
        if l_dark != -1 and r_dark != -1:
            break

        if l_dark == -1 and not is_line_light(img[y_start:y_end, a], standard):
            l_dark = a - 2 * width_step if a - 2 * width_step > x_start else x_start
        if r_dark == -1 and not is_line_light(img[y_start:y_end, (x_end + x_start - a - 1)], standard):
            next_right = x_end - a + 2 * width_step + x_start
            r_dark = next_right if next_right < x_end else x_end

    for b in range(y_start, y_end, height_step):
        if t_dark != -1 and b_dark != -1:
            break

        if t_dark == -1 and not is_line_light(img[b, l_dark:r_dark], standard):
            t_dark = b - 2 * height_step if b - 2 * height_step > y_start else y_start
        if b_dark == -1 and not is_line_light(img[y_end - b + y_start - 1, l_dark:r_dark], standard):
            next_b = y_end - b + y_start + 2 * height_step
            b_dark = next_b if next_b < y_end else y_end

    return t_dark, b_dark, l_dark, r_dark


def get_base_crop_area_cv2(img, standard):
    image_width = len(img[0, :])
    image_height = len(img[:, 0])
    t, b, l, r = get_crop_area_dark(img, standard, 0, image_width, 0, image_height,
                                    image_width // 100, image_height // 100)
    return get_crop_area_light(img, 95, l, r, t, b, (r - l) // 100, (b - t) // 100)


def get_sobel(channel):
    sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
    sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
    sobel = np.hypot(sobelx, sobely)

    return sobel


def find_significant_contours(img, sobel_8u):
    contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    level1 = []
    for i, tpl in enumerate(heirarchy[0]):
        if tpl[3] == -1:
            tpl = np.insert(tpl, 0, [i])
            level1.append(tpl)

    significant = []
    too_small = sobel_8u.size * 5 / 100
    for tpl in level1:

        contour = contours[tpl[0]]

        # epsilon = 0.10 * cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, 3, True)
        # contour = approx
        #
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
        if area > too_small:
            cv2.drawContours(img, [contour], 0, (255, 255, 255), 2, cv2.LINE_AA, maxLevel=1)
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant]


def segment(path_main, path_mask, name):

    img_mask = cv2.imread(path_mask)
    img_main = cv2.imread(path_main)

    image_width = len(img_mask[0, :])
    image_height = len(img_mask[:, 0])

    need_to_rotate = image_width > image_height

    t, b, l, r = get_base_crop_area_cv2(img_mask, 50)
    img_mask = img_mask[t:b, l:r]
    img_main = img_main[t:b, l:r]

    blurred = cv2.GaussianBlur(img_mask, (5, 5), 0)  # Remove noise

    sobel = np.max(np.array([get_sobel(blurred[:, :, 0]), get_sobel(blurred[:, :, 1]), get_sobel(blurred[:, :, 2])]),
                   axis=0)

    mean = np.mean(sobel)

    sobel[sobel <= mean] = 0
    sobel[sobel > 255] = 255

    sobel_8u = np.asarray(sobel, np.uint8)

    significant = find_significant_contours(img_main, sobel_8u)

    mask = sobel.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    mask = np.logical_not(mask)

    img_main[mask] = 255

    if need_to_rotate:
        img_main = rotate_image(img_main)

    fname = name.split('/')[-1]
    cv2.imwrite(output + '/' + fname, img_main)


def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE, image)


def crop_image(start_time, mask, main):
    print('start ' + main)
    segment(path_mask=mask, path_main=main, name=main)
    print('finished in: ' + str(time.time() - start_time))


def execute(folder):
    start_time = time.time()
    for pair in connect_mask_and_main(folder):
        mask, main = pair
        crop_image(start_time, mask, main)
        # p = multiprocessing.Process(target=crop_image, args=(start_time, mask, main))
        # p.start()


execute(first)
