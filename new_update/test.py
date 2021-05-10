import cv2
import face_utilities as fu
from imutils import face_utils
from face_utilities import Face_utilities
fu=Face_utilities()

cap = cv2.VideoCapture(0)


# def get_landmarks(self, frame, type):
#     '''
#     Get all facial landmarks in a face
#
#     Args:
#         frame (cv2 image): the original frame. In RGB format.
#         type (str): 5 or 68 facial landmarks
#
#     Outputs:
#         shape (array): facial landmarks' co-ords in format of of tuples (x,y)
#     '''
#     if self.predictor is None:
#         print("[INFO] load " + type + " facial landmarks model ...")
#         self.predictor = dlib.shape_predictor("../shape_predictor_" + type + "_face_landmarks.dat")
#         print("[INFO] Load model - DONE!")
#
#     if frame is None:
#         return None, None
#     # all face will be resized to a fix size, e.g width = 200
#     # face = imutils.resize(face, width=200)
#     # face must be gray
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = self.face_detection(frame)
#
#     if len(rects) < 0 or len(rects) == 0:
#         return None, None
#
#     shape = self.predictor(gray, rects[0])
#     shape = face_utils.shape_to_np(shape)
#
#     # in shape, there are 68 pairs of (x, y) carrying coords of 68 points.
#     # to draw landmarks, use: for (x, y) in shape: cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
#
#     return shape, rects
while True:
    ret, frame = cap.read()
    ret_process = fu.no_age_gender_face_process(frame, "68")
    if ret_process is None:
        continue
    rects, face, shape, aligned_face, aligned_shape = ret_process

    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("frame", frame)
cap.release()
cv2.destroyAllWindows()