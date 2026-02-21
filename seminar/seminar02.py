import numpy as np

def count_black_pixels(image):
    return np.sum(image > 230)

def center_of_mass(image):
    y_coords, x_coords = np.where(image > 230)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    return np.mean(x_coords), np.mean(y_coords)

def vertical_symmetry(image):
    left_half = image[:, :14]  
    right_half = np.fliplr(image[:, 14:])  
    return np.sum(left_half == right_half) / left_half.size  

def horizontal_symmetry(image):
    top_half = image[:14, :]  
    bottom_half = np.flipud(image[14:, :])  
    return np.sum(top_half == bottom_half) / top_half.size  

def count_horizontal_vertical_lines(image):
    center_row = image[14, :]  
    center_col = image[:, 14]  
    return np.sum(center_row > 230), np.sum(center_col > 230)  

def black_pixel_distribution(image):
    top_half = image[:8, :]  
    upper_middle = image[8:14, :]  
    lower_middle = image[14:20, :]
    bottom_half = image[20:, :]  
    return np.sum(top_half > 230), np.sum(upper_middle > 230), np.sum(lower_middle > 230), np.sum(bottom_half > 230)

def left_right_balance(image):
    left_half = image[:, :14]  
    right_half = image[:, 14:]  
    return np.sum(left_half > 230) - np.sum(right_half > 230)  

def count_center_crossings(image):
    center_row = image[14, :] > 230  
    center_col = image[:, 14] > 230  
    row_crossings = np.sum(np.diff(center_row.astype(int)) != 0)  
    col_crossings = np.sum(np.diff(center_col.astype(int)) != 0)  
    return row_crossings, col_crossings

def image_area(image):
    return np.sum(image > 230) / image.size  

def aspect_ratio(image):
    height, width = image.shape
    return height / width

def classify_digit(image):
    black_pixels = count_black_pixels(image)
    center = center_of_mass(image)
    vert_symmetry = vertical_symmetry(image)
    hor_symmetry = horizontal_symmetry(image)
    h_line, v_line = count_horizontal_vertical_lines(image)
    top_pixels, upper_middle_pixels, lower_middle_pixels, bottom_pixels = black_pixel_distribution(image)
    balance = left_right_balance(image)
    row_crossings, col_crossings = count_center_crossings(image)
    
    area = image_area(image)
    ratio = aspect_ratio(image)

    if center is not None:
        center_x, center_y = center
    else:
        center_x, center_y = 0, 0  

    if black_pixels < 50:
        return 1
    elif vert_symmetry > 0.92 and hor_symmetry > 0.92:
        return 0
    elif vert_symmetry > 0.85:
        return 8
    elif v_line > 30 and h_line < 5:
        return 1
    elif black_pixels > 430:
        return 0
    elif balance > 85:
        return 7
    elif balance < -85:
        return 4
    elif bottom_pixels > top_pixels + 70:
        return 4
    elif top_pixels > bottom_pixels + 70:
        return 9
    elif lower_middle_pixels > top_pixels and lower_middle_pixels > bottom_pixels:
        return 6
    elif upper_middle_pixels > lower_middle_pixels and upper_middle_pixels > bottom_pixels:
        return 3
    elif top_pixels > upper_middle_pixels and bottom_pixels > lower_middle_pixels:
        return 5
    elif row_crossings > 6:
        return 8
    elif col_crossings > 6:
        return 4

    if top_pixels > lower_middle_pixels and bottom_pixels < 40:
        return 3
    if row_crossings > 5 and lower_middle_pixels > top_pixels:
        return 3
    
    if vert_symmetry > 0.85 and hor_symmetry > 0.85 and upper_middle_pixels > 50 and lower_middle_pixels > 50:
        return 8
    if bottom_pixels > 100 and upper_middle_pixels > 80:
        return 8

    if area < 0.1 and ratio < 1.5:
        return 1
    if area > 0.3 and ratio > 1.5:
        return 8
    
    return np.random.choice([2, 3, 5])

if __name__ == "__main__":
    data = np.load("mnist.npz")
    x_test, y_test = data["x_test"], data["y_test"]
    
    predictions = []
    for i, img in enumerate(x_test):  
        pred = classify_digit(img)  
        predictions.append(pred)
    
    accuracy = np.mean(np.array(predictions) == y_test)
    print(f"Точность классификатора на всех примерах: {accuracy * 100:.2f}%")
