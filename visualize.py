import ast
import cv2
import numpy as np
import pandas as pd

print("1 - Imported libraries")


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # -- top-left
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # -- bottom-left
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # -- top-right
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # -- bottom-right
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


print("2 - Defined draw_border function")

results = pd.read_csv('./test_interpolated.csv')
print("3 - Loaded CSV file")

video_path = 'sample.mp4'
cap = cv2.VideoCapture(video_path)
print("4 - VideoCapture initialized")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))

print(f"5 - Video properties -> FPS: {fps}, Width: {width}, Height: {height}")

license_plate = {}
unique_car_ids = np.unique(results['car_id'])
print(f"6 - Found {len(unique_car_ids)} unique car IDs")

for car_id in unique_car_ids:
    print(f"7 - Processing car_id: {car_id}")
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])

    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': results[
            (results['car_id'] == car_id) &
            (results['license_number_score'] == max_)
            ]['license_number'].iloc[0]
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, results[
        (results['car_id'] == car_id) &
        (results['license_number_score'] == max_)
        ]['frame_nmr'].iloc[0])

    ret, frame = cap.read()
    print(f"8 - Read frame for car_id: {car_id}, Success: {ret}")

    bbox_str = results[
        (results['car_id'] == car_id) &
        (results['license_number_score'] == max_)
        ]['license_plate_bbox'].iloc[0]

    x1, y1, x2, y2 = ast.literal_eval(
        bbox_str.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
    license_plate[car_id]['license_crop'] = license_crop

print("9 - All license plates cropped and stored")

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print("10 - Reset frame pointer to beginning")

ret = True
while ret:
    ret, frame = cap.read()
    frame_nmr += 1
    if not ret:
        print(f"11 - End of video reached at frame {frame_nmr}")
        break

    df_ = results[results['frame_nmr'] == frame_nmr]
    print(f"12 - Processing frame number: {frame_nmr}, Detected cars: {len(df_)}")

    for row_indx in range(len(df_)):
        print(f"13 - Drawing bounding boxes for car #{row_indx}")
        car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
            df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25, line_length_x=200,
                    line_length_y=200)

        x1, y1, x2, y2 = ast.literal_eval(
            df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(
                ' ', ','))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

        license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']
        H, W, _ = license_crop.shape

        try:
            print(f"14 - Placing license crop and number for car_id: {df_.iloc[row_indx]['car_id']}")
            frame[int(car_y1) - H - 100:int(car_y1) - 100,
            int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

            frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
            int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

            (text_width, text_height), _ = cv2.getTextSize(
                license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                cv2.FONT_HERSHEY_SIMPLEX,
                4.3,
                17)

            cv2.putText(frame,
                        license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                        (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        (0, 0, 0),
                        17)
        except Exception as e:
            print(f"15 - Error placing license plate: {e}")

    out.write(frame)
    print(f"16 - Wrote frame {frame_nmr} to output video")

    frame = cv2.resize(frame, (1280, 720))
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)

print("17 - Releasing resources")
out.release()
cap.release()
print("18 - Done")
