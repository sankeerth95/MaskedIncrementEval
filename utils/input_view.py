import cv2
import torch
import cv2


def show_tensor_image(x: torch.Tensor):
    # visualize the zeroth bin only

    numpy_image = x[:, :, 0].cpu().numpy()
    cv2.imshow('a',cv2.Mat(numpy_image))
    cv2.waitKey(100)



def show_two_tensor_image(x1: torch.Tensor, x2:torch.Tensor):
    # visualize the zeroth bin only
    x = torch.cat([torch.sum(x1, dim=0), torch.sum(x2, dim=0)], dim=1)
    numpy_image = x.cpu().numpy()
    cv2.imshow('a',cv2.Mat(numpy_image))
    cv2.waitKey(10)


