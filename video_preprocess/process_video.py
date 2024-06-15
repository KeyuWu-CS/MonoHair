import cv2
import os

def ReadVideo(video_path, save_root, first_frame=False,save=True):
    print(save_root)
    if not os.path.exists(save_root):
        print(save_root)
        os.makedirs(save_root)

    cap = cv2.VideoCapture(video_path)
    i = 0
    frames= []
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if first_frame:
                cv2.imwrite(os.path.join(save_root, 'origin.png'), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            elif save:

                cv2.imwrite(os.path.join(save_root, '{}.png'.format(i)), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                frames.append(frame)

        # q键退出
        if not ret or first_frame:
            break
        i += 1
    cap.release()

    if not save:
        return frames





if __name__ == '__main__':

    interval=10  #### set interval to make the finally images around 300-600 frames, more frames, results more better when run colmap
    file = 'short_straight'
    root = '/datasets/wky/data/mvs_hair/video'
    video_root = root+'/{}.MP4'.format(file)
    save_root = root+r'/{}/capture_images'.format(file)
    frames = ReadVideo(video_root, save_root,save=False)


    image_root = root+'/{}/capture_images'.format(file)
    save_root = root+r'/{}/colmap/images'.format(file)
    os.makedirs(save_root, exist_ok=True)
    max_sharpless = 0
    for i, frame in enumerate(frames):
        img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
        if imageVar > max_sharpless:
            max_frame = frame
            max_sharpless = imageVar
            max_i = i
        if (i + 1) % interval == 0:
            max_sharpless = 0
            cv2.imwrite(os.path.join(save_root, '{}.png'.format(max_i)), max_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

