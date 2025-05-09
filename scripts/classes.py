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

        self.model = self._load_model()
    
    def _load_model(self):
        model = SimpleCNN()
        model.load_state_dict(torch.load('./model/weights/CNN.pth', weights_only=True))
        return model

    def _image2binary(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        min_value = np.min(gray_image)
        max_value = np.max(gray_image)
        #
        self.threshold = min(np.std(gray_image/255)/(max_value-min_value) * 650, 0.33)
        self.threshold = max(self.threshold, 0.26)
        #
        treshhold_value = min_value + (max_value - min_value)*self.threshold
        binary_image = (gray_image < treshhold_value).astype(np.uint8)
        return binary_image

    def _find_contour(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        #epsilon = 0.02 * cv2.arcLength(max_contour, True)
        max_contour = cv2.approxPolyN(max_contour, nsides=4, epsilon_percentage=-1, ensure_convex=True) #, epsilon_percentage=epsilon

        left_bottom_corner_id = np.argmin(np.sum((np.array([0,0])-max_contour[0])**2, axis=1))
        contour_ids = [(left_bottom_corner_id + i) % 4 for i in range(4)]
        max_contour = max_contour[0,contour_ids,:].reshape(1,4,2)

        #max_contour = cv2.approxPolyDP(max_contour, epsilon, True) # Ramer–Douglas–Peucker algorithm
        return max_contour
    
    def _square_field_transform(self, image, contour):
        square = np.array([[0, 0], [self.field_width, 0], [self.field_width, self.field_width], [0, self.field_width], ], dtype=np.float32)
        self.H, _ = cv2.findHomography(contour, square)
        output_field = cv2.warpPerspective(image, self.H, (self.field_width, self.field_width))
        output_field = output_field[int(self.field_width//9*0.25):int(self.field_width*(1-0.25/9)), int(self.field_width//9*0.25):int(self.field_width*(1-0.25/9))]
        output_field = cv2.resize(output_field, (self.field_width, self.field_width))
        return output_field
    
    def _cut_on_patches(self, image):
        width, height, channels = image.shape
        assert width%9==0 and height%9==0, f'{width%9}, {height%9}'
        array_of_values = image.reshape(9, width//9, 9, height//9, 3).transpose(0,2,1,3,4).reshape(-1, width//9, height//9, 3)
        return array_of_values

    def _patches2tensor(self, patches):
        patches = np.mean(patches/255, axis=-1) # rgb2gray
        h = (self.field_width//9)
        dh = h//10 # 10% of patch
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
        normalized_tensor = (binary_tensor - binary_tensor.mean()) / (binary_tensor.std() + 1e-5)

        return normalized_tensor, nonzeroes_rate
    
    def _reconstract(self, patches):
        reconstructed_field = patches.reshape(9, 9, self.field_width//9, self.field_width//9, 3).transpose(0, 2, 1, 3, 4).reshape(self.field_width, self.field_width, 3)

        for x in range(self.field_width//9, self.field_width, self.field_width//9):
            reconstructed_field[:, x-self.line_thick//2:x+self.line_thick//2, :] = (0, 255, 0)
        for y in range(self.field_width//9, self.field_width, self.field_width//9):
            reconstructed_field[y-self.line_thick//2:y+self.line_thick//2, :, :] = (0, 255, 0)

        return reconstructed_field

    def run(self, frame):
        binary_image = self._image2binary(frame)     
        contour = self._find_contour(binary_image)

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
            return cv2.drawContours(bin_im, contour, -1, (0, 255, 0), 2)
        
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
        cutted_patches = self._cut_on_patches(field)
        if(self.mode=='homography'):
            return self._reconstract(cutted_patches)
        
        patches_tensor, nonzeroes_rate = self._patches2tensor(cutted_patches)
        if(self.mode=='tensor'):
            return patches_tensor.numpy().reshape(9,9,1,self.patch_size,self.patch_size).transpose(0,3,1,4,2).reshape(self.patch_size*9, self.patch_size*9)
        
        model_output = self.model(patches_tensor[nonzeroes_rate >= self.nonzeros_treshold])
        prediction = torch.zeros(81, dtype=torch.long)
        prediction[nonzeroes_rate >= self.nonzeros_treshold] = (model_output.argmax(-1) + 1)
        
        if(self.mode=='predictions'):  
            rec_p = self._reconstract(cutted_patches)
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
            rec_p = self._reconstract(cutted_patches)
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
            #print(mask.shape, frame.shape)
            frame[mask] = output_field[mask]
            return frame
        
        return self._reconstract(cutted_patches)