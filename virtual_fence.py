# Đoạn mã AR (augmented reality - thực tế tăng cường) vẽ hàng rào ảo
# hình bán cầu sử dụng các thuật toán phát hiện và đối sánh các đặc
# trưng (SIFT và BF matcher) kết hợp với các phép biến đổi hình học
# trong không gian 3 chiều thực và 2 chiều của ảnh.

import numpy as np
import cv2 as cv

class RectSelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None

    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y])
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if self.drag_start:
            if flags & cv.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = self.drag_rect = None
                if rect:
                    self.callback(rect)

    def draw(self, vis):
        if self.drag_rect:
            x0, y0, x1, y1 = self.drag_rect
            cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

    @property
    def dragging(self):
        return self.drag_rect is not None

class PlaneTracker:
    def __init__(self):
        self.detector = cv.SIFT_create()
        self.matcher = cv.BFMatcher(cv.NORM_L2)
        self.targets = []

    def add_target(self, image, rect):
        x0, y0, x1, y1 = rect
        raw_points, raw_descrs = self.detect_features(image)
        points, descs = [], []
        for kp, desc in zip(raw_points, raw_descrs):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)
        descs = np.array(descs, dtype=np.float32)
        self.matcher.add([descs])
        self.targets.append(dict(image=image, rect=rect, keypoints=points, descrs=descs))

    def track(self, frame):
        frame_points, frame_descrs = self.detect_features(frame)
        if len(frame_points) < 10:
            return []
        matches = self.matcher.knnMatch(frame_descrs, k=2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < 10:
            return []
        matches_by_id = [[] for _ in range(len(self.targets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < 10:
                continue
            target = self.targets[imgIdx]
            p0 = [target['keypoints'][m.trainIdx].pt for m in matches]
            p1 = [frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, status = cv.findHomography(p0, p1, cv.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < 10:
                continue
            p0, p1 = p0[status], p1[status]
            x0, y0, x1, y1 = target['rect']
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)
            tracked.append(dict(target=target, p0=p0, p1=p1, H=H, quad=quad))
        tracked.sort(key=lambda t: len(t['p0']), reverse=True)
        return tracked

    def detect_features(self, frame):
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        return keypoints, descrs if descrs is not None else []

def create_capture(source=0):
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        print(f'Warning: unable to open video source: {source}')
    return cap

class App:
    def __init__(self, src):
        self.cap = create_capture(src)
        self.frame = None
        self.paused = False
        self.tracker = PlaneTracker()
        cv.namedWindow('plane')
        self.rect_sel = RectSelector('plane', self.on_rect)

    def on_rect(self, rect):
        self.tracker.add_target(self.frame, rect)

    def run(self):
        while True:
            playing = not self.paused and not self.rect_sel.dragging
            if playing or self.frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame = frame.copy()
            vis = self.frame.copy()
            if playing:
                tracked = self.tracker.track(self.frame)
                for tr in tracked:
                    cv.polylines(vis, [np.int32(tr['quad'])], True, (255, 255, 255), 2)
                    for (x, y) in np.int32(tr['p1']):
                        cv.circle(vis, (x, y), 2, (255, 255, 255))
                    self.draw_overlay(vis, tr)
            self.rect_sel.draw(vis)
            cv.imshow('plane', vis)
            ch = cv.waitKey(1)
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.tracker.targets = []
                self.tracker.matcher.clear()
            if ch == 27:
                break

    def draw_overlay(self, vis, tracked):
        x0, y0, x1, y1 = tracked['target']['rect']
        quad_3d = np.float32([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]])
        fx = 1.0
        h, w = vis.shape[:2]
        K = np.float64([[fx * w, 0, 0.5 * (w - 1)],
                        [0, fx * w, 0.5 * (h - 1)],
                        [0.0, 0.0, 1.0]])
        dist_coef = np.zeros(4)
        _, rvec, tvec = cv.solvePnP(quad_3d, tracked['quad'], K, dist_coef)

        hemisphere_verts = []
        num_lat, num_lon = 10, 20
        radius = (x1 - x0) * 1
        center_x, center_y, center_z = (x0 + x1) * 0.5, (y0 + y1) * 0.5, 0

        for i in range(num_lat + 1):
            lat = (np.pi / 2) * i / num_lat
            for j in range(num_lon + 1):
                lon = 2 * np.pi * j / num_lon
                x = center_x - radius * np.sin(lat) * np.cos(lon)
                y = center_y - radius * np.sin(lat) * np.sin(lon)
                z = center_z - radius * np.cos(lat)
                hemisphere_verts.append([x, y, z])

        hemisphere_verts = np.array(hemisphere_verts, dtype=np.float32)
        verts = cv.projectPoints(hemisphere_verts, rvec, tvec, K, dist_coef)[0].reshape(-1, 2)

        for i in range(num_lat):
            for j in range(num_lon):
                pt1 = verts[i * (num_lon + 1) + j]
                pt2 = verts[i * (num_lon + 1) + (j + 1) % (num_lon + 1)]
                pt3 = verts[(i + 1) * (num_lon + 1) + j]
                pt4 = verts[(i + 1) * (num_lon + 1) + (j + 1) % (num_lon + 1)]
                cv.line(vis, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 0, 255), 1)
                cv.line(vis, (int(pt1[0]), int(pt1[1])), (int(pt3[0]), int(pt3[1])), (0, 0, 255), 1)
                cv.line(vis, (int(pt2[0]), int(pt2[1])), (int(pt4[0]), int(pt4[1])), (0, 0, 255), 1)
                cv.line(vis, (int(pt3[0]), int(pt3[1])), (int(pt4[0]), int(pt4[1])), (0, 0, 255), 1)

if __name__ == '__main__':
    video_src = 2
    App(video_src).run()
    cv.destroyAllWindows()
