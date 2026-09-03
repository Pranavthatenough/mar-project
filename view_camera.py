
import cv2
import sys

def view_camera(idx=0):
    print(f"--- ATTEMPTING TO OPEN CAMERA {idx} ---")
    
    # Try multiple common Windows backends
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    cap = None
    for b in backends:
        print(f"Trying backend {b}...")
        cap = cv2.VideoCapture(idx, b)
        if cap.isOpened():
            print(f"SUCCESS: Opened camera {idx} with backend {b}")
            break
        cap.release()
    
    if not cap or not cap.isOpened():
        print(f"CRITICAL: Could not open camera {idx} with any backend.")
        return

    cv2.namedWindow("Camera View", cv2.WINDOW_NORMAL)
    print("Window created. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break
            
        cv2.putText(frame, f"Camera Index: {idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera View", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    view_camera(idx)
