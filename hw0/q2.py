import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def main(argv):
	img = Image.open(argv[0])
	ans = img.rotate(180)
	ans.save('ans2.png')

if __name__ == '__main__':
	main(sys.argv[1:])