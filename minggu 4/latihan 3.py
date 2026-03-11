import cv2
import numpy as np

class RealTimeEnhancement:

    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.history_buffer = []

    def enhance_frame(self, frame, enhancement_type='adaptive'):
        """
        Enhance single frame
        
        Parameters:
        frame : input video frame
        enhancement_type : 'adaptive', 'denoise', 'contrast'
        
        Returns:
        enhanced frame
        """

        # convert ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # pilih metode enhancement
        if enhancement_type == 'adaptive':
            enhanced = cv2.equalizeHist(gray)

        elif enhancement_type == 'denoise':
            enhanced = cv2.GaussianBlur(gray, (5,5), 0)

        elif enhancement_type == 'contrast':
            enhanced = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        else:
            enhanced = gray

        # simpan history untuk temporal smoothing
        self.history_buffer.append(enhanced)

        if len(self.history_buffer) > 5:
            self.history_buffer.pop(0)

        smoothed = np.mean(self.history_buffer, axis=0).astype(np.uint8)

        return smoothed


# =====================
# MAIN PROGRAM
# =====================

cap = cv2.VideoCapture(0)

enhancer = RealTimeEnhancement()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    enhanced_frame = enhancer.enhance_frame(frame, enhancement_type='adaptive')

    cv2.imshow("Original Video", frame)
    cv2.imshow("Enhanced Video", enhanced_frame)

    # tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()