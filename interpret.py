import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import ConvNet

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
	transforms.Grayscale(),
    transforms.ToTensor(),
	transforms.Normalize((0,1), (1,))
])

def image_loader(image):
	image = TRANSFORM(image).float()
	# image = torch.tensor(image, requires_grad=True)
	image = image.clone().detach().requires_grad_(True)
	image = image.unsqueeze(0)
	return image


if __name__ == "__main__":

	model = ConvNet()
	model.load_state_dict(torch.load("./neuralnet7"))
	model.eval()

	cv2.namedWindow("preview")
	vc = cv2.VideoCapture(0)

	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False

	while rval:

		frame = Image.fromarray(frame)
		frame = frame.crop(((frame.width - frame.height) // 2, 0,
							(frame.width + frame.height) // 2, frame.height))

		x = image_loader(frame)
		_, predicted = torch.max(model(x).data, 1)

		classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
				   'S', 'T', 'U', 'V', 'W', 'X', 'Y']

		print(classes[predicted.item()])

		cv2.imshow("preview", np.array(frame))
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27:
			break

	cv2.destroyWindow("preview")