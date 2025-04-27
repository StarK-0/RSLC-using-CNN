import cv2
import os

label = input("Enter label name: ")  # e.g. "hello"
save_dir = f"E:/Final Year Project/Final Review and Report/signlanguage/dataset/{label}"

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print(f"Starting image capture for label: {label}")

count = 0
while count < 50:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Capture - Press 's' to save", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):  # Press 's' to save image
        img_path = os.path.join(save_dir, f"{count+1}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        count += 1
    elif key & 0xFF == ord('q'):  # Press 'q' to quit early
        break

cap.release()
cv2.destroyAllWindows()
