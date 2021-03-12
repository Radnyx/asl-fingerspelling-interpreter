import cv2
from model import ConvNet

TRANSFORM = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(64),
    transforms.ToTensor()
])

def image_loader(loader, image):
	image = TRANSFORM(image).float()
	image = torch.tensor(image, requires_grad=True)
	image = image.unsqueeze(0)
	return image

if __name__ == "main":

	model m = ConvNet()
	model.load_state_dict(torch.load("./neuralnet"))
	model.eval()

	cv2.namedWindow("preview")
	vc = cv2.VideoCapture(0)

	if vc.isOpened():
		rval, frame = vc.read()
	else:
		rval = False

	while rval:

		x = image_loader(frame)
		print(np.argmax(model(x)).detach().numpy())

		cv2.imshow("preview", frame)
		rval, frame = vc.read()
		key = cv2.waitKey(20)
		if key == 27:
			break

	cv2.destroyWindow("preview")