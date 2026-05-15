import os
import cv2
import time

def main():
    label = input("Enter the sign label (example: A, B, C, L, Y, I_love_you): ").strip()
    save_dir = os.path.join("dataset", label)
    os.makedirs(save_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(save_dir) if f.lower().endswith(".jpg")]
    count = len(existing_files)

    camera_index = 0
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Could not open webcam on camera index 0.")
        print("Try changing camera_index to 1 in the script.")
        return

    time.sleep(1.0)
    for _ in range(10):
        cap.read()

    shots_per_press = 100
    delay_between_shots = 0.15 
    box_size = 300

    print("\nControls:")
    print("  t = capture 1 test image")
    print("  s = capture 100 images")
    print("  q = quit\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Could not read frame from webcam.")
            break

        display = frame.copy()
        h, w, _ = display.shape

        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"Label: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Saved: {count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Press 't' for 1 test shot", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "Press 's' for 100 shots", (20, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "Press 'q' to quit", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("ASL Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            roi = frame[y1:y2, x1:x2]
            filename = os.path.join(save_dir, f"{label}_{count}.jpg")
            cv2.imwrite(filename, roi)
            print(f"Saved test image: {filename}")
            count += 1

        elif key == ord('s'):
            print(f"Capturing {shots_per_press} images for label '{label}'...")

            for i in range(shots_per_press):
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Could not read frame during capture.")
                    break

                roi = frame[y1:y2, x1:x2]
                filename = os.path.join(save_dir, f"{label}_{count}.jpg")
                cv2.imwrite(filename, roi)
                count += 1

                preview = frame.copy()
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(preview, f"Capturing {i+1}/{shots_per_press}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("ASL Capture", preview)
                cv2.waitKey(1)

                time.sleep(delay_between_shots)

            print("Done capturing.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()