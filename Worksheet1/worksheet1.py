from numpy import expand_dims, asarrayfrom PIL import Imageimport random# load the imageimage_name = "dog.jpg"image = Image.open(image_name)print(image.size)print(type(image.size))data = asarray(image)print(data.shape)# Select random square patch patchdef select_random_patch(width, height, square_side):    random_coordinates = random.randint(0, height), random.randint(0, width)    # Prove if random patch can be constructed downwards (cannot be constructed downwards)    if height - random_coordinates[1] < square_side:        # Prove if random patch can be constructed upwards (cannot be constructed upwards)        if random_coordinates[1] < square_side:            return select_random_patch(width, height, square_side)        # Can be constructed upwards        else:            # Prove if random patch can be constructed to the right (cannot be constructed to the right)            if width - random_coordinates[0] < square_side:                # Prove if random patch can be constructed to the left (cannot be constructed to the left)                if random_coordinates[0] < 250:                    return select_random_patch(width, height, square_side)                # Can be constructed to the left. Random coordinates represent the bottom left corner of the random patch                else:                    right_lower = random_coordinates[0], random_coordinates[1]                    left_upper = random_coordinates[0] - square_side, random_coordinates[1] - square_side                    return left_upper, right_lower            # Can be constructed to the right. Random coordinates represent the bottom right corner of the random patch            else:                right_lower = random_coordinates[0] + square_side, random_coordinates[1]                left_upper = random_coordinates[0], random_coordinates[1] - square_side                return left_upper, right_lower    # Can be constructed downwards    else:        # Prove if random patch can be constructed to the right (cannot be constructed to the right)        if width - random_coordinates[0] < square_side:            # Prove if random patch can be constructed to the left (cannot be constructed to the left)            if random_coordinates[0] < 250:                return select_random_patch(width, height, square_side)            # Can be constructed to the left. Random coordinates represent the top right corner of the random patch            else:                right_lower = random_coordinates[0] - square_side, random_coordinates[1]                left_upper = random_coordinates[0], random_coordinates[1] - square_side                return left_upper, right_lower        # Can be constructed to the right. Random coordinates represent the top left corner of the random patch        else:            right_lower = random_coordinates[0] + square_side, random_coordinates[1] + square_side            left_upper = random_coordinates[0], random_coordinates[1]            return left_upper, right_lower