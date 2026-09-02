import math
import numpy as np
import torch
from PyQt6 import QtWidgets
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from scipy import ndimage

# define color for dots
dot_colors = ["lightblue", "cyan", "darkcyan", "lightgreen", "yellow"]
# origin_color = ["gold", "gold", "gold", "gold", "gold"]

def draw_all_points(painter, points, size=5, color=dot_colors):
    for i in range(5):
        if not np.any(np.isnan(points[i][0])):
            point = QPointF(points[i][0][0], points[i][0][1])
            painter.setPen(QPen(QColor(color[i]), size))
            painter.setBrush(QBrush(QColor(color[i])))
            painter.drawEllipse(point, size, size)

def find_circle_center(mask, layer, filter=0.1):
    # Filter the mask
    filtered_mask = mask[layer] >= filter

    # Label connected components
    labeled_mask, num_labels = ndimage.label(filtered_mask)
    # calc if have more than 1 label in layer
    if (num_labels > 0):
        # Find the largest component
        label_counts = np.bincount(labeled_mask.flat)[1:]
        largest_component_label = np.argmax(label_counts) + 1

        # Extract the largest component
        filtered_mask = (labeled_mask == largest_component_label)

    # Calculate the center of the largest component
    filtered_indices = np.where(filtered_mask)
    mean_x = np.mean(filtered_indices[1])
    mean_y = np.mean(filtered_indices[0])

    return (mean_x, mean_y)
# def find_circle_center(mask, layer, filter=0.5):
#     # Find the indices of the non-zero elements in the mask.
#     filtered_indices = mask[layer] >= filter
#     indicies = torch.nonzero(filtered_indices)
#     # Calculate the mean of the non-zero indices.
#     mean_x = indicies[:, 1].float().mean()
#     mean_y = indicies[:, 0].float().mean()

#     # Return the center of the circle.
#     return (mean_x.item(), mean_y.item())

def get_centers(pred_mask):
    pred_center = []
    # pred_center2 = []
    for i in range(5):
        pred_center.append([find_circle_center(pred_mask, i, 0.5), i+1])
        # pred_center2.append([find_circle_center2(pred_mask, i, 0.5), i+1])

    return pred_center#,pred_center2

def point_validation(points):
    for point in points:
        if np.any(np.isnan(point[0])) :
            print(f"Data contains missing points (NaN) at point {point[1]}")
            return False

    x = 0
    y = 1
    if min(point[0][x] for point in points) != points[0][0][x] : 
        print("Point 1 isn't the most left point")
        return False
    if min(point[0][y] for point in points) != points[4][0][y] : 
        print("Point 5 isn't the top point")
        return False
    if max(point[0][y] for point in points) != points[2][0][y] : 
        print("Point 3 isn't the lowest point")
        return False
    if max(point[0][x] for point in points) != points[2][0][x] and max(point[0][x] for point in points) != points[4][0][x] : 
        print("Point 3 or 5 isn't the rightmost point")
        return False

    if points[1][0][x] > points[3][0][x] : 
        print("Point 2 is at the right of point 4")
        return False
    if points[1][0][y] > points[3][0][y] : 
        print("Point 2 is below of point 4")
        return False
    if points[3][0][x] > points[2][0][x] : 
        print("Point 4 is at right of point 3")
        return False
    if points[3][0][x] > points[4][0][x] : 
        print("Point 4 is at right of point 5")
        return False

    # 1-2連線與水平交角需< +-30度
    slope_of_degree30 = (math.tan(math.pi * (30/180)))
    slope_line_12 = (points[1][0][y]-points[0][0][y])/(points[1][0][x]-points[0][0][x])
    if slope_line_12 > slope_of_degree30 or slope_line_12 < -slope_of_degree30:
        print("slope of line 1 and 2 is larger then 30 degree")
        return False

    return True


def draw_line(painter, point1, point2, color, size=3):
    line = QLineF(point1[0], point1[1], point2[0], point2[1])
    # Create a line object.
    painter.setPen(QPen(QColor(color), size))
    painter.drawLine(line)
import numpy as np

def extend_line(point1, point2, image_size, edge=50):
    """
    Extends a line to stretch across the whole figure.

    Args:
        point1: The coordinates of the first point.
        point2: The coordinates of the second point.
        image_size: The size of the image.

    Returns:
        The coordinates of the extended line.
    """

    # check leftmost point
    if point1[0] > point2[0]:
        point1, point2 = point2, point1

    # Calculate the slope of the line.
    if point2[0] - point1[0] == 0:
        return [point1[0], edge], [point2[0], image_size[1]-edge]
    elif point2[1] - point1[1] == 0:
        return [edge, point1[1]], [image_size[0]-edge, point2[1]]

    slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
    # Calculate the y-intercept of the line.
    y_intercept = point1[1] - slope * point1[0]

    # calculate extented y using y = ax+b, a = slope, b = y_intercept
    p1y = edge*slope + y_intercept
    p2y = (image_size[0]-edge)*slope + y_intercept

    p1x, p2x = edge, image_size[0]-edge
    
    # check leftmost boiundary
    if p1y > image_size[1] or p1y < 0:
        p1y = edge * (slope>0) + (image_size[1]-edge) * (slope < 0)
        p1x = (p1y - y_intercept) / slope

    # check rightmost boiundary
    if p2y > image_size[1] or p2y < 0:
        p2y = edge * (slope<0) + (image_size[1]-edge) * (slope > 0)
        p2x = (p2y - y_intercept) / slope
        
    return [p1x, p1y], [p2x, p2y]

def slope_calculation(point1, point2):
    """
    Calculate the slope of 2 given points

    Args:
        point1: The coordinates of the first point.
        point2: The coordinates of the second point.

    Returns:
        The slope of the line form by the 2 given points.
    """
    return (point2[1]-point1[1])/(point2[0]-point1[0])

def draw_all_line(painter, points, SIZE):
    # calculate all line's extend points and draw the line
    p1, p2 = extend_line(points[0][0], points[1][0], SIZE)
    draw_line(painter, p1, p2, "red")
    p1, p2 = extend_line(points[2][0], points[3][0], SIZE)
    draw_line(painter, p1, p2, "green")
    p1, p2 = extend_line(points[3][0], points[4][0], SIZE)
    draw_line(painter, p1, p2, "blue")
    # draw all points
    draw_all_points(painter, points)

def angle_between_lines(line1point1, line1point2, line2point1, line2point2):
    """
    Calculates the angle between two lines given two sets of points.

    Args:
        line1point1: point 1 of line 1 in format of [x1, y1].
        line1point2: point 2 of line 1 in format of [x2, y2].
        line2point1: point 1 of line 2 in format of [x1, y1].
        line2point2: point 2 of line 2 in format of [x2, y2].

    Returns:
        The angle between the two lines in radians.
    """

    # Get direction vectors
    vec1 = [line1point2[0] - line1point1[0], line1point2[1] - line1point1[1]]
    vec2 = [line2point2[0] - line2point1[0], line2point2[1] - line2point1[1]]

    # Calculate dot product and magnitudes
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v**2 for v in vec1))
    magnitude2 = math.sqrt(sum(v**2 for v in vec2))

    # Prevent division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    # Cosine of the angle
    cos_theta = dot_product / (magnitude1 * magnitude2)

    # Clamp to avoid acos errors
    cos_theta = max(-1.0, min(1.0, cos_theta))

    # Angle in radians
    theta = math.acos(cos_theta) * (180 / math.pi)

    if theta > 90 : theta = 180 - theta

    return theta 

def calculate_alpha_beta(points):
    alpha = angle_between_lines(points[0][0], points[1][0], points[2][0], points[3][0])
    beta = angle_between_lines(points[0][0], points[1][0], points[3][0], points[4][0])

    if alpha < 50: print(f"發育不良, alpha角為{alpha}度")
    elif alpha < 60: print(f"追蹤再檢, alpha角為{alpha}度")

    if beta > 77: print(f"半脫位或脫位, beta角為{beta}")
    elif beta >= 55: print(f"追蹤再檢, beta角為{beta}")

    return alpha, beta

