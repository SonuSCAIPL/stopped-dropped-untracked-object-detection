import argparse
import cv2
import time


bg_image = cv2.imread("742.jpg")
bg_image = cv2.resize(bg_image, (640, 360))

top, bottom =80,300

def get_opencv_result(video_to_process):
    # create VideoCapture object for further video processing
    captured_video = cv2.VideoCapture(video_to_process)
    # check video capture status
    if not captured_video.isOpened:
        print("Unable to open: " + video_to_process)
        exit(0)

    # instantiate background subtraction
    background_subtr_method = cv2.bgsegm.createBackgroundSubtractorGSOC()

    start_time = time.time()
    while True:
        # read video frames
        retval, frame = captured_video.read()
        # check whether the frames have been grabbed
        if not retval:
            break
        # resize video frames
        frame = cv2.resize(frame, (640, 360))
       
        # pass the frame to the background subtractor
        foreground_mask = background_subtr_method.apply(frame, learningRate=-1)
        # obtain the background without foreground mask
        background_img = background_subtr_method.getBackgroundImage()
        
        end_time =time.time()
        if end_time != start_time:
            fps=1/(end_time-start_time)
            cv2.putText(frame,"FPS:"+str(int(fps)), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        start_time = time.time()
    
        diff = cv2.absdiff(background_img, bg_image) 
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 70, 1000, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=5)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if top < y < bottom:
            
                if cv2.contourArea(contour) < 500:
                    continue
                cv2.rectangle(background_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
        #cv2.drawContours(background_img, contours, -1, (0, 255, 0), 2)
       
        # show the current frame, foreground mask, subtracted result
        cv2.imshow("Initial Frames", frame)
        cv2.imshow("Foreground Masks", foreground_mask)
        cv2.imshow("Subtraction Result", background_img)
        cv2.imshow("Subtraction2", dilated)
      
        keyboard = cv2.waitKey(1)
        if keyboard == 27:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video",
        type=str,
        help="Define the full input video path",
        default="space_traffic.mp4",
    )

    # parse script arguments
    args = parser.parse_args()

    # start BS-pipeline
    get_opencv_result(args.input_video)