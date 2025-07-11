import glob 
import cv2
import os

# Media creation functions
def create_video_from_images(image_dir):
    images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    if not images:
        print(f"No images found in {image_dir}")
        return
        
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(image_dir, 'evolution.mp4')
    out = cv2.VideoWriter(video_path, fourcc, 8.0, (width, height))
    
    for image in images:
        frame = cv2.imread(image)
        out.write(frame)
    
    out.release()


def create_all_videos(dir_list):
    for dir_name in dir_list:
        if os.path.exists(dir_name):
            create_video_from_images(dir_name)


# File management functions
def clear_dirs(dir_list):
    for dir_name in dir_list:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        for file in glob.glob(os.path.join(dir_name, '*')):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing {file}: {e}")