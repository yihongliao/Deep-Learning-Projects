from PIL import Image
import torchvision.transforms as tvt
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

im = []
tensorImg = []
hist = []

# open images as PIL objects
im.append(Image.open("orange.jpg"))
im.append(Image.open("orange2.jpg"))

# compose transforms
xform2 = tvt.Compose([tvt.ToTensor(), tvt.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

hist.clear()
for i in range(2):

	tmp_hist = []

	# convert PIL objects to tensors
	tensorImg.append(xform2(im[i]))

	# calculate histograms for images
	for j in range(3):

		tmp_hist.append(torch.histc(tensorImg[i][j], bins = 20, min = -1.0, max = 1.0)) 

		### Normalizing the channel based hists so that the bin counts in each sum to 1.
		tmp_hist[j] = tmp_hist[j].div(tmp_hist[j].sum())

	hist.append(tmp_hist)

	# plot the histograms before the transforms
	x = np.arange (-0.95, 1.05, 0.1)

	plt.figure(i)
	plt.subplot(4, 1, 1)
	plt.imshow(im[i])

	for j in range(3):
		plt.subplot(4, 1, j+2)
		plt.bar(x, hist[i][j], 0.07, align='center')
		plt.xticks(np.arange(-1, 1, step=0.1))

	plt.xlabel('Bins')

# calculate the distance between histograms
for j in range(3):
	dist = wasserstein_distance( torch.squeeze( hist[0][j] ).cpu().numpy(), torch.squeeze( hist[1][j] ).cpu().numpy() )
	print("\n Wasserstein distance for channel: ", dist)

#########################################################################################################################
# apply affine transforms
unnormalizer = tvt.Normalize([-1, -1, -1],[2, 2, 2])

hist.clear()
for i in range(2):

	tmp_hist = []

	# apply affine transform
	affine_imgs = tvt.functional.affine(tensorImg[i], angle=20, translate=(0.2, 0.1), scale=1.75, shear=0)
	
	# calculate histograms for images
	for j in range(3):

		tmp_hist.append(torch.histc(affine_imgs[j], bins = 20, min = -1.0, max = 1.0)) 

		### Normalizing the channel based hists so that the bin counts in each sum to 1.
		tmp_hist[j] = tmp_hist[j].div(tmp_hist[j].sum())

	hist.append(tmp_hist)

	# plot the histograms after the affine transform
	x = np.arange (-0.95, 1.05, 0.1)

	plt.figure(i+2)
	plt.subplot(4, 1, 1)
	plt.imshow(unnormalizer(affine_imgs).permute(1, 2, 0))

	for j in range(3):
		plt.subplot(4, 1, j+2)
		plt.bar(x, hist[i][j], 0.07, align='center')
		plt.xticks(np.arange(-1, 1, step=0.1))

	plt.xlabel('Bins')

# calculate the distance between histograms after the affine transform
for j in range(3):
	dist = wasserstein_distance( torch.squeeze( hist[0][j] ).cpu().numpy(), torch.squeeze( hist[1][j] ).cpu().numpy() )
	print("\n Wasserstein distance for channel after the affine transform: ", dist)

#########################################################################################################################
# apply perspective transforms
perspective_transformer = tvt.RandomPerspective(distortion_scale=0.6, p=1.0)

hist.clear()
for i in range(2):

	tmp_hist = []

	# apply affine transform
	perspective_imgs = perspective_transformer(tensorImg[i])
	
	# calculate histograms for images
	for j in range(3):

		tmp_hist.append(torch.histc(perspective_imgs[j], bins = 20, min = -1.0, max = 1.0)) 

		### Normalizing the channel based hists so that the bin counts in each sum to 1.
		tmp_hist[j] = tmp_hist[j].div(tmp_hist[j].sum())

	hist.append(tmp_hist)

	# plot the histograms after the affine transform
	x = np.arange (-0.95, 1.05, 0.1)

	plt.figure(i+4)
	plt.subplot(4, 1, 1)
	plt.imshow(unnormalizer(perspective_imgs).permute(1, 2, 0))

	for j in range(3):
		plt.subplot(4, 1, j+2)
		plt.bar(x, hist[i][j], 0.07, align='center')
		plt.xticks(np.arange(-1, 1, step=0.1))

	plt.xlabel('Bins')

# calculate the distance between histograms after the perspective transform
for j in range(3):
	dist = wasserstein_distance( torch.squeeze( hist[0][j] ).cpu().numpy(), torch.squeeze( hist[1][j] ).cpu().numpy() )

	print("\n Wasserstein distance for channel after the perspective transform: ", dist)

plt.show()