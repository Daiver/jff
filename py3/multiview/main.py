import json
import numpy as np
from pathlib import Path
import cv2
from scipy.optimize import minimize
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


root = Path("/home/daiver/Downloads/views_d18_0/")


scale_factor = 0.1


def downscale(img):
    return cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)


def read_landmarks(fname):
    with open(fname) as f:
        data = json.load(f)
    print(data)
    points = [
        data["LeftEyeCorners"][0:2],
        data["LeftEyeCorners"][2:4],

        data["RightEyeCorners"][0:2],
        data["RightEyeCorners"][2:4],

        # data["LeftEyeCenters"][0:2],
        # data["LeftEyeCenters"][2:4],
        # data["RightEyeCenters"][0:2],
        # data["RightEyeCenters"][2:4],

        data["MouthCenters"][0:2],
        data["MouthCenters"][2:4],
        data["MouthCorners"][0:2],
        data["MouthCorners"][2:4],
    ]
    points = np.array(points, dtype=np.float32)
    points *= scale_factor
    return points


def draw_landmarks(canvas, landmarks, color=(0, 255, 0)):
    assert landmarks.shape[1] == 2
    for l in landmarks:
        p = (int(round(l[0])), int(round(l[1])))
        cv2.circle(canvas, p, 3, color, -1)


def project_landmarks(landmarks_positions, cameras_parameters):
    n_images = cameras_parameters.shape[0]
    ones = torch.ones(len(landmarks_positions), 1)
    landmarks_positions = torch.cat((landmarks_positions, ones), dim=1)

    projected_landmarks_per_image = []
    for img_ind in range(n_images):
        params_raw = cameras_parameters[img_ind]
        projection_matrix = torch.cat((params_raw, torch.FloatTensor([1])))
        projection_matrix = projection_matrix.view(3, 4)

        projected_landmarks = (projection_matrix @ landmarks_positions.transpose(0, 1)).transpose(0, 1)
        projected_landmarks = projected_landmarks[:, 0:2] / projected_landmarks[:, 2].view(-1, 1)
        # projected_landmarks = projected_landmarks[:, 0:2]
        projected_landmarks_per_image.append(projected_landmarks)

    return torch.stack(projected_landmarks_per_image)


def loss_function(landmarks_positions, cameras_parameters, target_landmark_positions_per_image):
    """
    :param landmarks_positions: n_landmarks x 3
    :param cameras_parameters: n_images x 11
    :param target_landmark_positions_per_image: n_images x n_landmarks x 2
    :return:
    """

    n_images = cameras_parameters.shape[0]

    projected_landmarks_per_image = project_landmarks(landmarks_positions, cameras_parameters)
    loss = 0.0
    for img_ind in range(n_images):
        residuals = projected_landmarks_per_image[img_ind] - target_landmark_positions_per_image[img_ind]
        loss = loss + torch.norm(residuals)
    return loss


def main():
    img1 = downscale(cv2.imread(str(root / "0.jpg")))
    img2 = downscale(cv2.imread(str(root / "1_2.jpg")))

    landmarks1 = read_landmarks(str(root / "0.json"))
    landmarks2 = read_landmarks(str(root / "1_2.json"))

    # landmarks = torch.randn(8, 3)
    # landmarks = torch.randn(8, 3)
    # cameras_parameters = torch.randn(2, 11)

    target_landmarks_per_image = torch.FloatTensor(np.array([landmarks1, landmarks2]))
    landmarks = torch.ones(target_landmarks_per_image.size(1), 3) * 0.1
    # landmarks = torch.FloatTensor([
    #     [-2, 0, 0],
    #     [-1, 0, 0],
    #     [1, 0, 0],
    #     [2, 0, 0],
    #
    #     [0, -1, 0],
    #     [0, -3, 0],
    #     [-1, -2, 0],
    #     [1, -2, 0]
    # ])
    # cameras_params = torch.ones(2, 11)
    cameras_params = torch.FloatTensor([
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1,
        ],
        [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1,
        ]
    ])

    landmarks.requires_grad_(True)
    # cameras_params.requires_grad_(True)

    for outer_iter, lr in enumerate([1e-3, 9e-4, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]):
        if outer_iter > 2:
            cameras_params.requires_grad_(True)
        optimizer = optim.Adam([landmarks, cameras_params], lr=lr)
        for i in range(4000):
            loss = loss_function(landmarks, cameras_params, target_landmarks_per_image)
            print(f"{i} -> {loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    projected_landmarks_per_image = project_landmarks(landmarks, cameras_params)
    print(projected_landmarks_per_image)

    print(target_landmarks_per_image)
    # optimization_result = minimize(func, x0, jac=jac)
    # print(optimization_result)

    landmarks1_p = projected_landmarks_per_image[0].detach().numpy()
    landmarks2_p = projected_landmarks_per_image[1].detach().numpy()
    landmarks = landmarks.detach()

    draw_landmarks(img1, landmarks1)
    draw_landmarks(img1, landmarks1_p, (255, 0, 0))

    draw_landmarks(img2, landmarks2)
    draw_landmarks(img2, landmarks2_p, (255, 0, 0))

    cv2.imshow("1", img1)
    cv2.imshow("2", img2)
    cv2.waitKey()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(landmarks[:, 0].numpy(), landmarks[:, 1].numpy(), landmarks[:, 2].numpy(), 'ro')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
