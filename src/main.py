import cv2
import model as m
import gestures as g

cap = cv2.VideoCapture(0)   # manages the connection/resource to webcam

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# load memes
angrybird = cv2.imread("assets/angrybirds.png")
dog = cv2.imread("assets/dog-hat.png")

# state variables
current_meme = None
prev_nose_x = None

# windows
cv2.namedWindow("Webcam")
cv2.namedWindow("Meme")

while True:
    #Capture frame by frame
    ret, frame = cap.read() # ret - boolean that indicates whether a frame was successfully read from video source
                            # frame - actual image data returned as NumPy array 
    frame = cv2.flip(frame, 1)

    # converts raw webcam frame to model ready tensor
    input_tensor = m.preprocess_frame(frame)

    # feed tensor into model input slot
    m.interpreter.set_tensor(m.input_index, input_tensor)
    
    # runs inference (model processes the image)
    m.interpreter.invoke()
    
    # gets model output (pose keypoints)
    output = m.interpreter.get_tensor(m.output_index)

    keypoints = output[0][0]    # removes extra dimensions
                                # each row is now [keypoint_index][x, y, confidence]

    
    
    # joints
    nose = keypoints[0]
    left_wrist = keypoints[9]


    # angry birds meme
    if g.left_hand_left_ear(keypoints):
        current_meme = angrybird
    
    #dog
    elif nose[2] > 0.5:
        current_x = nose[1]

        if g.head_move_left(prev_nose_x, current_x):
            current_meme = dog
        prev_nose_x = current_x 
    


    # show webcam
    cv2.imshow("Webcam", frame)
    
    # show meme window
    if current_meme is not None:
        cv2.imshow("Meme", current_meme)
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):  # pauses for 1 millisecond to wait for keypress
        break



cap.release() # closes video file
cv2.destroyAllWindows() # closes all windows 