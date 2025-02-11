def extractor(image):
    # Extract the first channel of the depth image
    image = image[:, :, 0]  
    print(f"reshaped image: {image.shape}")

    return image

if __name__ == "__main__":
    import cv2
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Process an image file.")
    parser.add_argument("image_path", type=str, help="Relative or absolute path to the image.")

    args = parser.parse_args()

    print(args.image_path)

    img = cv2.imread(args.image_path)
    
    image = extractor(img)

    # Save the new image to the same folder but with new name
    filename = "new_depth.png"
    output_dir = "/home/philiph/Documents/PhiliphExjobb/automatic_scene_reconstruction/SAM-6D/SAM-6D/Data/Example"

    os.chdir(output_dir)
    cv2.imwrite(filename, image)