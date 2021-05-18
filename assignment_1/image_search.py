import os
import csv
from pathlib import Path
import argparse

import cv2

OUTPATH = 'output'

# when given a small tuple of (filename, distance) this will return the distance to sort by
def sort_by_distance(tuple):
	return tuple[1]

# make sure we only load valid image files
# omitting target_name will just check for file extensions
def valid_image(file, target_name = ''):
	file = file.lower()
	target_name = target_name.lower()

	valid_extensions = (
		'.jpg',
		'.jpeg',
		'.bmp',
		'.png',
		'.webp',
		'.tif',
		'.tiff'
	)

	ext = os.path.splitext(file)[1]
	ext_valid = ext in valid_extensions

	# file is not valid if it is the target image or if it does not have a correct image extension
	return file != target_name and ext_valid

# creates a normalized histogram with all 3 color channels
def color_histogram(image, normalize_function = cv2.NORM_MINMAX):
	hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
	normalized = cv2.normalize(hist, hist, 0, 255, normalize_function)
	return normalized


def main(target_name, data_path):
	# generates paths and names
	target_path = os.path.join(data_path, target_name) # ./data/image.jpg
	target_basename = os.path.splitext(target_name)[0] # image

	# filter off invalid files to compare with
	collection = [file for file in os.listdir(data_path) if valid_image(file, target_name)]

	# what to expect
	collection_len = len(collection)
	print(f'There are {collection_len} images to compare with in this collection.')
	
	# create headers for the csv file
	# output = [('filename', 'distance')]
	output = []

	# load target image and create histogram
	target_image = cv2.imread(target_path)
	target_hist = color_histogram(target_image)
	
	# indices are just used for the progress bar
	for i, filename in enumerate(collection):
		filepath = os.path.join(data_path, filename)
		
		# load the comparison image and create histogram
		comparison_image = cv2.imread(filepath)
		comparison_hist = color_histogram(comparison_image)
		
		# this is the similarity value
		chisqr = cv2.compareHist(target_hist, comparison_hist, cv2.HISTCMP_CHISQR)
		
		# add to output list
		output.append((filename, round(chisqr, 2))) # one pair of parens is a tuple here

		# print the actual progress for every 10%
		if i % (collection_len // 10) == 0:
			print(f'{int((i + 1) / (collection_len // 10) * 10)}% done')

	# order by CHISQR distance - low to high
	output.sort(key=sort_by_distance)

	# Most similar image is first in the sorted list
	print(f"The image most similar to {target_name} is {output[0][0]} with a CHISQR distance of {output[0][1]}")

	# add csv headers
	output.insert(0, ('filename', 'distance'))

	# write csv to the given path
	outfile = os.path.join(OUTPATH, f'{target_basename}.csv')
	with open(outfile, 'w', encoding='utf-8') as fh:
		csv.writer(fh).writerows(output)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='generate histogram comparisons between a target image and a collection of images')
	parser.add_argument('target_name', help='the name of the image to compare the collection to (inside of data_path)')
	parser.add_argument("-d", "--data_path", default=Path('./data/'), type = Path, help = 'the path to the directory containing the images to compare to the target image')
	args = parser.parse_args()

	main(args.target_name, args.data_path)