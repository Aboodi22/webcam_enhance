"""
Live webcam enhancer (OpenCV).

Install:   pip install opencv-python numpy
Optional:  pip install pyvirtualcam   (to send the result to Zoom/Meet/Teams via OBS Virtual Camera)

Keys (click on the video window first):
  q  quit
  1  toggle white balance
  2  toggle contrast (CLAHE)
  3  toggle gamma / brightness
  4  toggle denoise
  5  toggle sharpen
  + / -  raise / lower gamma brightness
  s  save a snapshot
  c  switch to next camera index
"""
import time
import cv2
import numpy as np

CAMERA_INDEX = 0          # try 1 if the wrong camera opens
WIDTH, HEIGHT, FPS = 1280, 720, 30
USE_VIRTUAL_CAM = False   # set True after installing pyvirtualcam + OBS

state = {"wb": True, "clahe": True, "gamma": True, "denoise": True, "sharpen": True}
gamma_value = 1.2         # >1 brightens, <1 darkens

clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    return cap


def gray_world_wb(img):
    """Simple auto white balance: make the average of each channel equal."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(img)]
    mb, mg, mr = b.mean(), g.mean(), r.mean()
    avg = (mb + mg + mr) / 3
    b *= avg / (mb + 1e-6)
    g *= avg / (mg + 1e-6)
    r *= avg / (mr + 1e-6)
    return cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)


def apply_clahe(img):
    """Local contrast boost on the lightness channel only (keeps colors natural)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe_op.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def apply_gamma(img, g):
    inv = 1.0 / g
    table = ((np.arange(256) / 255.0) ** inv * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def denoise(img):
    # Bilateral filter keeps edges, smooths noise. Fast enough for live video.
    return cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)


def sharpen(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def enhance(frame):
    out = frame
    if state["denoise"]:
        out = denoise(out)
    if state["wb"]:
        out = gray_world_wb(out)
    if state["clahe"]:
        out = apply_clahe(out)
    if state["gamma"]:
        out = apply_gamma(out, gamma_value)
    if state["sharpen"]:
        out = sharpen(out)
    return out


def main():
    global gamma_value
    cam_index = CAMERA_INDEX
    cap = open_camera(cam_index)
    if not cap.isOpened():
        print("Could not open the camera. Close other apps using it, or change CAMERA_INDEX.")
        return

    vcam = None
    if USE_VIRTUAL_CAM:
        import pyvirtualcam
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vcam = pyvirtualcam.Camera(width=w, height=h, fps=FPS, fmt=pyvirtualcam.PixelFormat.BGR)
        print("Virtual camera:", vcam.device)

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame received.")
            break

        result = enhance(frame)

        now = time.time()
        fps = 1 / max(now - prev, 1e-6)
        prev = now

        if vcam is not None:
            vcam.send(result)
            vcam.sleep_until_next_frame()

        # Side-by-side: original | enhanced
        view = np.hstack([frame, result])
        view = cv2.resize(view, None, fx=0.6, fy=0.6)
        cv2.putText(view, f"{fps:.0f} FPS  gamma={gamma_value:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Original (left) | Enhanced (right)", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"): state["wb"] = not state["wb"]
        elif key == ord("2"): state["clahe"] = not state["clahe"]
        elif key == ord("3"): state["gamma"] = not state["gamma"]
        elif key == ord("4"): state["denoise"] = not state["denoise"]
        elif key == ord("5"): state["sharpen"] = not state["sharpen"]
        elif key in (ord("+"), ord("=")): gamma_value = min(3.0, gamma_value + 0.1)
        elif key == ord("-"): gamma_value = max(0.3, gamma_value - 0.1)
        elif key == ord("s"):
            name = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(name, result)
            print("Saved", name)
        elif key == ord("c"):
            cap.release()
            cam_index += 1
            cap = open_camera(cam_index)
            if not cap.isOpened():
                cam_index = 0
                cap = open_camera(cam_index)

    cap.release()
    if vcam is not None:
        vcam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
