import cv2
import numpy as np
np.random.seed(42)
# import the necessary packages
import torch.nn.functional as F
import argparse
import torch

from generate_dataset import *

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to the trained PyTorch model")
args = vars(ap.parse_args())

# set the device we will be using to test the model
dev = "cpu"
if torch.cuda.is_available():
    dev = "cuda"
print(dev)
device = torch.device(dev)

# load the model and set it to evaluation mode
model = torch.load(args["model"],map_location=torch.device(dev)).to(device)
model.eval()

classes = ["ellipse","rectangle","triangle"]

total, errors = 0, 0 # for errors calculs

# switch off autograd
with torch.no_grad():
    # loop over the test set
    while True:
            total += 1
            # generate a random image
            img, gtLabel = generate_random()

            # Convert NumPy array to tensor and resize it
            # image = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            # image /= 255.0
            # image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=True)
            image = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
            gray_img = torch.mean(image, dim=1, keepdim=True)
            gray_img /= 255.0
            image = F.interpolate(gray_img, size=(64, 64), mode='bilinear', align_corners=True)


            # send image to model for prediction
            image = image.to(device)
            pred = model(image)

            # find the class label index with the largest corresponding probability
            idx = pred.argmax(axis=1).cpu().numpy()[0]
            predLabel = classes[idx]

            # draw the predicted class label on the image
            color = (0, 255, 0) if gtLabel == predLabel else (0, 0, 255)
            if gtLabel != predLabel:
                errors += 1

            cv2.putText(img, gtLabel, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

            # display the result in terminal and show the input image
            print("[INFO] ground truth label: {}, predicted label: {}".format(gtLabel, predLabel))
            cv2.imshow("image", img)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('q'):
                break
            else:
                cv2.destroyAllWindows()

percent = "{:.2f}".format(100.0 - ((errors*100)/total))
print(f"Total images generated : {total}\nTotal prediction errors : {errors}\nTotal accuracy : {errors}/{total} = {percent}%")

    
