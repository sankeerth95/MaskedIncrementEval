import cv2
import torch
import cv2

def show_tensor_image(x1: torch.Tensor, x2:torch.Tensor):
    # visualize the zeroth bin only
    x = torch.cat([x1[0], x2[0]], dim=1)
    numpy_image = x.cpu().numpy()
    cv2.imshow('a',cv2.Mat(numpy_image))
    cv2.waitKey(10)


