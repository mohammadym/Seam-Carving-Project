from __future__ import print_function
import argparse
import numba
import numpy as np
from PIL import Image
from tqdm import trange
from energy_functions import dual_gradient_energy
import cv2


def energy_map(img, fn):
    x0 = np.roll(img, -1, axis=1).T
    x1 = np.roll(img, 1, axis=1).T
    return fn(x0, x1).T

@numba.jit()
def cumulative_energy(energy):
    height, width = energy.shape
    paths = np.zeros((height, width), dtype=np.int64)
    path_energies = np.zeros((height, width), dtype=np.int64)
    path_energies[0] = energy[0]
    paths[0] = np.arange(width) * np.nan
    for i in range(1, height):
        for j in range(width):
            prev_energies = path_energies[i-1, max(j-1, 0):j+2]
            least_energy = prev_energies.min()
            path_energies[i][j] = energy[i][j] + least_energy
            paths[i][j] = np.where(prev_energies == least_energy)[0][0] - (1*(j != 0))
    return paths, path_energies


def seam_end(energy_totals):
    return list(energy_totals[-1]).index(min(energy_totals[-1]))


def find_seam(paths, end_x):
    height, width = paths.shape[:2]
    seam = [end_x]
    for i in range(height-1, 0, -1):
        cur_x = seam[-1]
        offset_of_prev_x = paths[i][cur_x]
        seam.append(cur_x + offset_of_prev_x)
    seam.reverse()
    return seam


def remove_seam(img, seam):
    height, width = img.shape[:2]
    return np.array([np.delete(img[row], seam[row], axis=0) for row in range(height)])


def resize_image(full_img, cropped_pixels, energy_fn, address, num):
    img = full_img.copy()
    img_copy = full_img.copy()
    seam_list = []
    img_list = []
    if num == 0:
        for i in trange(cropped_pixels, desc='cropping image by {0} calumns'.format(cropped_pixels)):
            e_map = energy_map(img, energy_fn)
            e_paths, e_totals = cumulative_energy(e_map)
            seam = find_seam(e_paths, seam_end(e_totals))
            seam_list.append(seam)
            img = remove_seam(img, seam)
    if num == 1:
        for i in trange(cropped_pixels, desc='cropping image by {0} rows'.format(cropped_pixels)):
            e_map = energy_map(img, energy_fn)
            e_paths, e_totals = cumulative_energy(e_map)
            seam = find_seam(e_paths, seam_end(e_totals))
            seam_list.append(seam)
            img = remove_seam(img, seam)
    drawn_img = draw_seam(img_copy, seam_list, address, num)
    img_list.append(drawn_img)
    img_list.append(img)
    return img_list


def draw_seam(img, seam_list, address, num):
    img = cv2.imread(address)
    if num ==0:
        for seam in seam_list:
            k = 0
            for i in seam:
                img[k, i] = (255, 0 , 0)
                k += 1
    if num == 1:
        for seam in seam_list:
            k = 0
            for i in seam:
                img[i, k] = (255, 0 , 0)
                k += 1
    return img


def main():
    rows_num = int(input("rows:"))
    cals_num = int(input("cals:"))
    parser = argparse.ArgumentParser(description="Intelligently crop an image along one axis")
    parser.add_argument('input_file')
    args = vars(parser.parse_args())
    print(args)
    num = 0
    format = args['input_file'].split('.')
    if format[1] == "png":
        temp_img = Image.open(args['input_file'])
        rgb_im = temp_img.convert('RGB')
        rgb_im.save('new.jpg')
        img = np.array(Image.open('new.jpg'))
    else:
        img = np.array(Image.open(args['input_file']))
    img_list = resize_image(img, cals_num, dual_gradient_energy, args['input_file'], num)
    cropped_img = img_list[1]
    img = np.transpose(cropped_img, axes=(1, 0, 2))
    num = 1
    img_lists = resize_image(img, rows_num, dual_gradient_energy, args['input_file'], num)
    cropped_img = img_lists[1]
    seam_img_1 = img_list[0]
    seam_img_3 = img_lists[0]
    final_img = np.transpose(cropped_img, axes=(1, 0, 2))
    seam_img_4 = np.transpose(seam_img_3, axes=(1, 0, 2))
    seam_img_5 = np.transpose(seam_img_4, axes=(1, 0, 2))
    Image.fromarray(final_img).save("imgs/out.jpg")
    Image.fromarray(seam_img_1).save("imgs/vertical_seams.jpg")
    Image.fromarray(seam_img_5).save("imgs/horizontal_seams.jpg")

if __name__ == "__main__":
    main()