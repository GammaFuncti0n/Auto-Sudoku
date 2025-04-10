import cv2
import numpy as np

class Camera():
    def __init__(self, camera_config):
        self.camera_id = camera_config['camera_id']
        self.window_size = camera_config['window_size']

    def create_capture(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        assert self.cap.isOpened(), "Can not open camera"

class FrameProcessor():
    def __init__(self, process_config):
        self.threshold = process_config['threshold']
        self.field_width = process_config['field_width']
        self.process_config = process_config
        self.line_thick = process_config['utils']['line_thick']
        self.mode = process_config['utils']['mode']

    def _image2binary(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        min_value = np.min(gray_image)
        max_value = np.max(gray_image)
        treshhold_value = min_value + (max_value - min_value)*self.threshold
        binary_image = (gray_image < treshhold_value).astype(np.uint8)
        return binary_image

    def _find_contour(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        #epsilon = 0.02 * cv2.arcLength(max_contour, True)
        max_contour = cv2.approxPolyN(max_contour, nsides=4, epsilon_percentage=-1, ensure_convex=True) #, epsilon_percentage=epsilon

        left_bottom_corner_id = np.argmin(np.sum((np.array([self.field_width,0])-max_contour[0])**2, axis=1))
        contour_ids = [(left_bottom_corner_id + i) % 4 for i in range(4)]
        max_contour = max_contour[0,contour_ids,:].reshape(1,4,2)

        #max_contour = cv2.approxPolyDP(max_contour, epsilon, True) # Ramer–Douglas–Peucker algorithm
        return max_contour
    
    def _square_field_transform(self, image, contour):
        square = np.array([[self.field_width, 0], [self.field_width, self.field_width], [0, self.field_width], [0, 0]], dtype=np.float32)
        H, _ = cv2.findHomography(contour, square)
        output_field = cv2.warpPerspective(image, H, (self.field_width, self.field_width))
        output_field = output_field[int(self.field_width//9*0.22):int(self.field_width*(1-0.22/9)), int(self.field_width//9*0.22):int(self.field_width*(1-0.22/9))]
        output_field = cv2.resize(output_field, (self.field_width, self.field_width))
        return output_field
    
    def _cut_on_patches(self, image):
        width, height, channels = image.shape
        assert width%9==0 and height%9==0, f'{width%9}, {height%9}'
        array_of_values = image.reshape(9, width//9, 9, height//9, 3).transpose(0,2,1,3,4).reshape(-1, width//9, height//9, 3)
        return array_of_values
    
    def _reconstract(self, patches):
        reconstructed_field = patches.reshape(9, 9, self.field_width//9, self.field_width//9, 3).transpose(0, 2, 1, 3, 4).reshape(self.field_width, self.field_width, 3)

        for x in range(self.field_width//9, self.field_width, self.field_width//9):
            reconstructed_field[:, x-self.line_thick//2:x+self.line_thick//2, :] = (0, 255, 0)
        for y in range(self.field_width//9, self.field_width, self.field_width//9):
            reconstructed_field[y-self.line_thick//2:y+self.line_thick//2, :, :] = (0, 255, 0)

        return reconstructed_field

        
    def run(self, frame):
        binary_image = self._image2binary(frame)
        if(self.mode=='binary'):
            return 255*binary_image
        
        contour = self._find_contour(binary_image)
        if(self.mode=='max_contour'):
            for i in range(4):
                cv2.putText(
                    frame,
                    text=f'{i+1}',
                    org=contour[0,i], 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=self.line_thick,
                    lineType=cv2.LINE_AA
                    )
            return cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)

        field = self._square_field_transform(frame, contour)
        cutted_field = self._cut_on_patches(field)
        return self._reconstract(cutted_field)