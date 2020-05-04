from numpy import expand_dims, asarrayimport numpy as npimport cv2from PIL import Imageimport randomrefPt = []def click_and_patching(event, image, x, y):    global refPt    if event == cv2.EVENT_LBUTTONDOWN:        refPt = [(x, y)]    elif event == cv2.EVENT_LBUTTONUP:        refPt.append((x, y))        cv2.rectangle(image, refPt[0], refPt[1], thickness=5)        cv2.rectangle(image, refPt[0], refPt[1], (0, 0, 0), thickness=5)        cv2.imshow("Image", image)# Select random square patch patchdef select_random_patch(width, height, square_side):    while True:        random_coordinates = abs(random.randint(0, width)), abs(random.randint(0, height))        # Prove if random patch can be constructed downwards (cannot be constructed downwards)        if abs(height - random_coordinates[1]) < square_side:            # Prove if random patch can be constructed upwards (cannot be constructed upwards)            if random_coordinates[1] < square_side:                pass            # Can be constructed upwards            else:                # Prove if random patch can be constructed to the right (cannot be constructed to the right)                if abs(width - random_coordinates[0]) < square_side:                    # Prove if random patch can be constructed to the left (cannot be constructed to the left)                    if random_coordinates[0] < 250:                        pass                    # Can be constructed to the left. Random coordinates represent the bottom left corner of the random patch                    else:                        right_lower = random_coordinates[0], random_coordinates[1]                        left_upper = random_coordinates[0] - square_side, random_coordinates[1] - square_side                        return left_upper, right_lower                # Can be constructed to the right. Random coordinates represent the bottom right corner of the random patch                else:                    right_lower = random_coordinates[0] + square_side, random_coordinates[1]                    left_upper = random_coordinates[0], random_coordinates[1] - square_side                    return left_upper, right_lower        # Can be constructed downwards        else:            # Prove if random patch can be constructed to the right (cannot be constructed to the right)            if abs(width - random_coordinates[0]) < square_side:                # Prove if random patch can be constructed to the left (cannot be constructed to the left)                if random_coordinates[0] < square_side:                    pass                # Can be constructed to the left. Random coordinates represent the top right corner of the random patch                else:                    right_lower = random_coordinates[0], random_coordinates[1] + square_side                    left_upper = random_coordinates[0] - square_side, random_coordinates[1]                    return left_upper, right_lower            # Can be constructed to the right. Random coordinates represent the top left corner of the random patch            else:                right_lower = random_coordinates[0] + square_side, random_coordinates[1] + square_side                left_upper = random_coordinates[0], random_coordinates[1]                return left_upper, right_lowerdef patch_in_image(left_upper_coordinate, background, patch):    background.paste(patch, left_upper_coordinate)    background.save("patchDog.jpg")