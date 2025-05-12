import cv2
from .classes import Camera, FrameProcessor

class Task():
    def __init__(self, config):
        self.config = config
        self.camera_config = config['camera']
        self.pocess_config = config['process']
    
    def run(self):
        camera = Camera(self.camera_config)
        camera.create_capture()

        frame_processor = FrameProcessor(self.pocess_config)

        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera', camera.window_size) 
        while True:
            ret, frame = camera.cap.read()
            assert ret, "Can not get frame"
            processed_frame = frame_processor.run(frame)
            cv2.imshow('Camera', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        camera.cap.release()
        cv2.destroyAllWindows()