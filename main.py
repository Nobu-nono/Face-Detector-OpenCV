import cv2 as cv

face_ref = cv.CascadeClassifier ('face_ref.xml')
camera = cv.VideoCapture (0)

def face_detector (frame) :
    gray = cv.cvtColor (frame, cv.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale (frame, scaleFactor=1.1, minNeighbors=3, minSize=(150, 150))
    return faces

def generate_box (frame, faces) :
    for x, y, w, h in faces :
        cv.rectangle (frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText (frame, "Face Detected", (x, y-10), 1, cv.FONT_HERSHEY_COMPLEX, (255, 255, 255), 1)

def exit_programe () :
    camera.release ()
    cv.destroyAllWindows () 
    exit ()

def main () :
    while True :
        _, frame = camera.read ()
        frame = cv.flip (frame, 1)
        faces = face_detector (frame)
        cv.putText (frame, f"Faces : {len(faces)}", (20, 40), 1, cv.FONT_HERSHEY_COMPLEX, (255, 255, 255), 1)
        generate_box (frame, faces)
        cv.imshow ("Face Detector", frame)
        if cv.waitKey (1) & 0xFF == 27 :
            exit_programe ()


if __name__ == '__main__' :
    main ()
