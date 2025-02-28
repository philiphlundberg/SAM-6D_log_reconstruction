def extractor(args, image):

    if args.img_type=="Depth":
        print(f"Depth image of shape {image.shape} detected")
        if len(image.shape)==3:
            # Extract the first channel of the depth image
            print("Extracting only one channel...")
            image = image[:, :, 0]
        else:
            print("No channel extracted")
            # exit the program without saving
            exit()
    elif args.img_type=="RGB":
        print(f"RGB image of shape {image.shape} detected")
        if image.shape[-1]==4:
            print("Removing alpha channel...")
            image = image[:, :, :3]
        else:
            print("No alpha channel removed")
            # exit the program without saving
            exit()   
    print(f"reshaped image: {image.shape}")
    return image

if __name__ == "__main__":
    import cv2
    import argparse
    import os
    import datetime

    parser = argparse.ArgumentParser(description="Process an image file.")
    parser.add_argument("--process", type=str, choices=["Check-Size", "Resize"], required=True, help="Specify what you want to do with the image")
    parser.add_argument("--img_type", type=str, choices=["Depth", "RGB"], help="Specify 'Depth' or 'RGB' image")
    parser.add_argument("--image_path", type=str, help="Relative or absolute path to the image.")
    parser.add_argument("--output_name", type=str, help="Desired output name of the file")

    args = parser.parse_args()

    # Get the directory the image is taken from
    if args.image_path:
        image_dir = os.path.dirname(os.path.abspath(args.image_path))
        print(f"Image taken from {image_dir}")
    else:
        image_dir = os.getcwd()

    print(args.image_path)

    img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Error: Could not load image from path: {args.image_path}")
    
    if args.process == "Check-Size":
        print(f"Image shape: {img.shape}")
        exit()
    elif args.process == "Resize":
        # Resize the image to 512x512
        image = extractor(args, img)  
      
        if args.output_name:
            filename = args.output_name
        else:
            # Save the new image to the same folder but with new name using current time
            filename = "depth_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png"

        output_path = os.path.join(image_dir, filename)
        cv2.imwrite(output_path, image)