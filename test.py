import os
from PIL import Image, ImageDraw, ImageFile  # Python image library

import time

MAIN_KEY = 'main'
MASK_KEY = 'mask'
LIGHT_PIXEL_PERCENTAGE = 90
ImageFile.LOAD_TRUNCATED_IMAGES = True
first = 'images/1'
output = 'images/result'

for pair in connect_mask_and_main(first):
    mask, main = pair

    print("working on mask - " + mask)
    start_time = time.time()
    mask_img = Image.open(mask)  # open mask photo
    main_img = Image.open(main)

    width, height = mask_img.size

    need_to_rotate = width > height

    start_light_crop = time.time()
    light_base_crop_area = get_base_crop_area(mask_img.load(), width, height, 50)

    mask_img = mask_img.crop(light_base_crop_area)
    main_img = main_img.crop(light_base_crop_area)

    square_length = 2
    mask_pixels = mask_img.load()  # take colour x,y point

    width, height = mask_img.size  # determine size of images
    draw = ImageDraw.Draw(main_img)
    white = (255, 255, 255)

    for z in range(width):
        for j in range(height):
            if sum(mask_pixels[z, j]) >= 500:
                draw.point((z, j), white)
                # and check_square_not_dark(mask_pixels, z, j, 3, 510, width, height):

    mask_img.close()
    if need_to_rotate:
        main_img = main_img.rotate(270, expand=True)  # rotate not for every photos

    main_img.save(os.path.join(output, os.path.basename(main)))
    print(main + ' - DONE')
    print("exec time = {}".format(str(time.time() - start_time)))
    main_img.close()
