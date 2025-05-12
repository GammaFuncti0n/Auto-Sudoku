import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from model import SimpleCNN

class Camera():
    def __init__(self, camera_config):
        self.camera_id = camera_config['camera_id']
        self.window_size = camera_config['window_size']

    def create_capture(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        assert self.cap.isOpened(), "Can not open camera"

class FrameProcessor():
    def __init__(self, process_config):
        self.process_config = process_config
        #self.threshold = process_config['threshold']
        self.field_width = process_config['field_width']  
        self.patch_size = process_config['patch_size']  
        self.nonzeros_treshold = process_config['nonzeros_treshold']
        self.contrast_treshold = process_config['contrast_treshold']
        self.line_thick = process_config['utils']['line_thick']
        self.mode = process_config['utils']['mode']

        self._load_model()
    
    def _load_model(self) -> None:
        """
        Create and load model weights.
        """
        self.model = SimpleCNN()
        self.model.load_state_dict(torch.load('./model/weights/CNN.pth', weights_only=True))

    def _image2binary(self, image) -> np.ndarray:
        """
        Convert input image to binary format with dynamic treshold.
        """
        # Range of values ​​from 0 to 255
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        min_value = np.min(gray_image)
        max_value = np.max(gray_image)
        
        # Dynamic selection of threshold based on MinMax values ​​and standard deviation of image pixels
        standart_deviation = np.std(gray_image/255)
        minmax_range = max_value - min_value
        threshold_value = standart_deviation / minmax_range * 650
        
        # Cliping treshold and applying it
        cliped_threshold = np.clip(threshold_value, 0.26, 0.33)
        self.threshold = min_value + (max_value - min_value) * cliped_threshold
        binary_image = (gray_image < self.threshold).astype(np.uint8)

        return binary_image

    def _find_max_contour(self, binary_image) -> None:
        """
        Find on binary image all contours and choose the max contour based on its closed area
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Approximate founded contour into rectangle
        max_contour = cv2.approxPolyN(max_contour, nsides=4, epsilon_percentage=-1, ensure_convex=True)
        #max_contour = cv2.approxPolyDP(max_contour, epsilon, True) # Ramer–Douglas–Peucker algorithm

        # Sort corners by clock-wise from top-left corner
        left_bottom_corner_id = np.argmin(np.sum((np.array([0,0])-max_contour[0])**2, axis=1))
        contour_ids = [(left_bottom_corner_id + i) % 4 for i in range(4)]

        self.field_contour = max_contour[0,contour_ids,:].reshape(1,4,2)
    
    def _homography_transform(self, image) -> np.ndarray:
        """
        Applaying homography transformation for transform sudoku field on image into the square
        """
        square = np.array([[0, 0], [self.field_width, 0], [self.field_width, self.field_width], [0, self.field_width], ], dtype=np.float32)
        self.H, _ = cv2.findHomography(self.field_contour, square)
        homography_field = cv2.warpPerspective(image, self.H, (self.field_width, self.field_width))

        # Truncate field 2.8% from each side
        i_1 = int(self.field_width * 0.25 / 9)
        i_2 = int(self.field_width * (1 - 0.25 / 9))
        truncated_field = homography_field[i_1:i_2, i_1:i_2]
        square_field = cv2.resize(truncated_field, (self.field_width, self.field_width))

        return square_field
    
    def _image2patches(self, image) -> np.ndarray:
        """
        Reshape input image to 81 rgb patches (9*9) 
        """
        width, height, _ = image.shape
        assert width%9==0 and height%9==0, f'{width%9}, {height%9}'
        patches = image.reshape(9, width//9, 9, height//9, 3).transpose(0,2,1,3,4).reshape(-1, width//9, height//9, 3)
        return patches
    
    def _patches2image(self, patches) -> np.ndarray:
        """
        Reshape rgb patches to image of field
        """
        image = patches.reshape(9, 9, self.field_width//9, self.field_width//9, 3).transpose(0, 2, 1, 3, 4).reshape(self.field_width, self.field_width, 3)
        return image

    def _patches2tensor(self, patches):
        patches = np.mean(patches/255, axis=-1) # rgb2gray
        h = (self.field_width//9)
        dh = int(h/100*5) # 5% of patch
        patches = patches[:,dh:h-dh,dh:h-dh]

        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.patch_size,self.patch_size)),
        ])
        tensor = 1 - torch.stack([transformation(img) for img in patches]).to(torch.float32)

        # Min-Max normalization
        mins = tensor.view(81, -1).min(dim=1)[0].view(-1, 1, 1, 1)
        maxs = tensor.view(81, -1).max(dim=1)[0].view(-1, 1, 1, 1)
        ranges = maxs - mins
        low_contrast_mask = (ranges < self.contrast_treshold)
        tensor = (tensor - mins) / (maxs - mins + 1e-8)

        # Zeros for empty patch
        replacement = torch.zeros_like(tensor)
        tensor = torch.where(low_contrast_mask.view(tensor.shape[0], 1, 1, 1), replacement, tensor)

        # Binarization patches and calculate rate of nonzeroes values in center
        binary_tensor = (tensor>0.5).to(torch.float32)
        nonzeroes_rate = binary_tensor[:,:,int(self.patch_size/3):int(2*self.patch_size/3),int(self.patch_size/3):int(2*self.patch_size/3)].mean((-1,-2,-3))
        binary_tensor = torch.where((nonzeroes_rate < self.nonzeros_treshold).view(tensor.shape[0], 1, 1, 1), replacement, binary_tensor)

        # Normal normalization
        normalized_tensor = (binary_tensor - binary_tensor.mean()) / (binary_tensor.std() + 1e-8)

        return normalized_tensor, nonzeroes_rate
    
    def _reconstract(self, patches):
        reconstructed_field = patches.reshape(9, 9, self.field_width//9, self.field_width//9, 3).transpose(0, 2, 1, 3, 4).reshape(self.field_width, self.field_width, 3)

        for x in range(self.field_width//9, self.field_width, self.field_width//9):
            reconstructed_field[:, x-self.line_thick//2:x+self.line_thick//2, :] = (0, 255, 0)
        for y in range(self.field_width//9, self.field_width, self.field_width//9):
            reconstructed_field[y-self.line_thick//2:y+self.line_thick//2, :, :] = (0, 255, 0)

        return reconstructed_field

    def run(self, frame) -> np.ndarray:
        """
        Main class method.
        Gets a frame from a video, applies all transformations and returns a modified frame for display on the screen.
        """
        binary_image = self._image2binary(frame)     
        self._find_max_contour(binary_image)

        if(self.mode=='binary'):
            bin_im = cv2.cvtColor(255*binary_image, cv2.COLOR_GRAY2RGB)
            cv2.putText(
                bin_im,
                text=f'{self.threshold:.3f}',
                org=[100,100], 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=self.line_thick,
                lineType=cv2.LINE_AA
                )
            return cv2.drawContours(bin_im, self.field_contour, -1, (0, 255, 0), 2)
        
        if(self.mode=='max_contour'):
            for i in range(4):
                cv2.putText(
                    frame,
                    text=f'{i+1}',
                    org=self.field_contour[0,i], 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=self.line_thick,
                    lineType=cv2.LINE_AA
                    )
            return cv2.drawContours(frame, self.field_contour, -1, (0, 255, 0), 2)

        square_field = self._homography_transform(frame)
        patches = self._image2patches(square_field)
        
        if(self.mode=='homography'):
            reconstructed_image =  self._patches2image(patches)

            for x in range(self.field_width//9, self.field_width, self.field_width//9):
                reconstructed_image[:, x-self.line_thick//2:x+self.line_thick//2, :] = (0, 255, 0)
            for y in range(self.field_width//9, self.field_width, self.field_width//9):
                reconstructed_image[y-self.line_thick//2:y+self.line_thick//2, :, :] = (0, 255, 0)
            
            return reconstructed_image
        
        patches_tensor, nonzeroes_rate = self._patches2tensor(patches)
        if(self.mode=='tensor'):
            return patches_tensor.numpy().reshape(9,9,1,self.patch_size,self.patch_size).transpose(0,3,1,4,2).reshape(self.patch_size*9, self.patch_size*9)
        
        model_output = self.model(patches_tensor[nonzeroes_rate >= self.nonzeros_treshold])
        prediction = torch.zeros(81, dtype=torch.long)
        prediction[nonzeroes_rate >= self.nonzeros_treshold] = (model_output.argmax(-1) + 1)
        
        if(self.mode=='predictions'):  
            rec_p = self._patches2image(patches)
            for i in range(81):
                if nonzeroes_rate[i] >= self.nonzeros_treshold:
                    text = f'{prediction[i]}'
                else:
                    text = '?'

                cv2.putText(
                    rec_p,
                    text=text,
                    org=[i//9*self.field_width//9+10, i%9*self.field_width//9+30], 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=self.line_thick,
                    lineType=cv2.LINE_AA
                    )
            return rec_p
        
        if(self.mode=='predictions_homography'):
            rec_p = self._patches2image(patches)
            for i in range(81):
                if nonzeroes_rate[i] >= self.nonzeros_treshold:
                    text = f'{prediction[i]}'
                else:
                    text = '?'

                cv2.putText(
                    rec_p,
                    text=text,
                    org=[i//9*self.field_width//9+10, i%9*self.field_width//9+30], 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 255, 0),
                    thickness=self.line_thick,
                    lineType=cv2.LINE_AA
                    )
            inverse_H = np.linalg.inv(self.H)
            output_field = cv2.warpPerspective(rec_p, inverse_H, (frame.shape[1], frame.shape[0]))
            mask = (output_field!=0)
            frame[mask] = output_field[mask]
            return frame
        
        return frame