import os
import cv2
import numpy as np
import PySimpleGUI as sg
import threading

def capture_frames(read_path, write_path, capture_interval_seconds=10, video_codec='.mp4'):
    # Read every .mp4 file in directory
    video_list = [f for f in os.listdir(read_path) if video_codec in f]

    def process_video(video_file):
        print(f"{video_file} 변환 시작")
        video_path = os.path.join(read_path, video_file)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(video_file + ' is not found')
            return

        # Calculate the frame interval according to the input time
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        capture_interval_frames = capture_interval_seconds * fps
        print(f"Capture interval: {capture_interval_frames} frames")

        # capturing frame
        count = 0
        num = 0
        while True:
            ret, frame = cap.read()
            count += 1
            if count % capture_interval_frames != 0:
                continue
            if ret:
                num += 1
                cv2.imwrite(os.path.join(write_path, f"{video_file[:-4]}_{num:03d}.png"), frame)
                print(f"{video_file[:-4]}_{num:03d}.png 저장완료")
            else:
                print("no frame!")
                break
        print(f"{video_file} 변환 완료")

    # Create and start threads for each video file
    threads = [threading.Thread(target=process_video, args=(video_file,)) for video_file in video_list]
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

def show_capture_frames_window():
    layout1 = [
        [sg.Text("video에서 img 추출", font=('Helvetica', 16))],
        [sg.Text("read_path:"), sg.Input(), sg.FolderBrowse()],
        [sg.Text("write_path:"), sg.Input(), sg.FolderBrowse()],
        [sg.Text("추출 간격 (초):"), sg.Input(default_text="10")],
        [sg.Button("추출")]
    ]
    window1 = sg.Window("chicken_finder", layout1)

    while True:
        event1, values1 = window1.read()

        if event1 == sg.WIN_CLOSED:
            break

        elif event1 == "추출":
            read_path = values1[0]
            write_path = values1[1]
            capture_interval_seconds = int(values1[2]) if values1[2].isdigit() else 10
            try:
                capture_frames(read_path, write_path, capture_interval_seconds=capture_interval_seconds)
                sg.popup("추출이 완료되었습니다!", title="완료")
            except:
                sg.popup("read_path와 write_path를 알맞게 설정하세요")

    window1.close()

def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def remove_white(read_path, thresh, num_skip_images):
    image_files = [f for f in os.listdir(read_path) if f.endswith(".png")]
    accumulated_result = cv2.imread(os.path.join(read_path, image_files[0]))

    layout_show_images = [
        [sg.Text(size=(40, 1), key='-IMAGE_NAME-')],
        [sg.Image(key="-IMAGE-")],
        [sg.Button("다음 이미지 보기"), sg.Button("저장"), sg.Input(default_text="1", size=(5, 1), key='-NUM_SKIP-')],
        [sg.Button("폐사체찾기"), sg.Text("Min Area:"), sg.Input(default_text="200", size=(5, 1), key='-MIN_AREA-'),
         sg.Text("Max Area:"), sg.Input(default_text="800", size=(5, 1), key='-MAX_AREA-')]
    ]
    window_show_images = sg.Window("폐사체 검출 결과", layout_show_images, finalize=True)

    current_image_idx = 0
    window_contour = None  # Initialize the contour window as None
    event3, values3 = None, None  # Initialize event3 and values3

    while True:
        if event3 != "폐사체찾기" and current_image_idx + num_skip_images < len(image_files):
            for _ in range(num_skip_images):
                next_img = cv2.imread(os.path.join(read_path, image_files[current_image_idx + 1]))
                gray_img = cv2.cvtColor(accumulated_result, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | thresh)
                mask_next = cv2.bitwise_not(mask)
                bg = np.zeros(accumulated_result.shape)
                img_with_hole = cv2.copyTo(accumulated_result, mask, bg)
                accumulated_result = cv2.copyTo(next_img, mask_next, img_with_hole)
                current_image_idx += 1

            resized_result = resize_image(accumulated_result, 25)
            img_bytes = cv2.imencode(".png", resized_result)[1].tobytes()
            window_show_images["-IMAGE-"].update(data=img_bytes)
            window_show_images["-IMAGE_NAME-"].update(image_files[current_image_idx])

        event3, values3 = window_show_images.read()

        if event3 == sg.WIN_CLOSED:
            break
        elif event3 == "다음 이미지 보기":
            if current_image_idx + num_skip_images >= len(image_files):  # Updated this line
                sg.popup('마지막 이미지입니다.')
            else:
                num_skip_images = int(values3['-NUM_SKIP-'])
        elif event3 == "저장":
            save_path = sg.popup_get_file('저장할 위치를 선택하세요.', save_as=True)
            if save_path:
                cv2.imwrite(save_path, accumulated_result)
                sg.popup('이미지 저장 완료!')
        elif event3 == "폐사체찾기":
            min_area = int(values3['-MIN_AREA-'])
            max_area = int(values3['-MAX_AREA-'])
            contour_img = find_contour(accumulated_result, min_area, max_area)
            contour_img_bytes = cv2.imencode(".png", contour_img)[1].tobytes()

            # New window for contour image
            if window_contour:  # Close the window if it already exists
                window_contour.close()
            layout_contour = [
                [sg.Image(key="-CONTOUR-")],
            ]
            window_contour = sg.Window("Contour Image", layout_contour, finalize=True)
            window_contour["-CONTOUR-"].update(data=contour_img_bytes)

    window_show_images.close()
    if window_contour:  # Close the contour window if it exists
        window_contour.close()

def find_contour(img, min_area, max_area):
    # preprocessing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0) # 파라미터로 놓을까나?

    _, img_thresh = cv2.threshold(img_blur, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size, aspect ratio, and solidity
    filtered_contours = [contour for contour in contours if is_valid_contour(contour, min_area, max_area)]

    cv2.drawContours(img, filtered_contours, -1, (0, 255, 0), 2)

    return resize_image(img, 25)

def is_valid_contour(contour, min_area, max_area):
    # Check contour size
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False

    # Check contour aspect ratio
    _, _, w, h = cv2.boundingRect(contour)
    if w > 2.5 * h or h > 2.5 * w:
        return False

    # Check contour solidity
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area
    if solidity < 0.8:
        return False
    return True

def show_remove_white_window():
    layout2 = [
        [sg.Text("폐사체 검출", font=('Helvetica', 16))],
        [sg.Text("read_path:"), sg.Input(), sg.FolderBrowse()],
        [sg.Text("thresh 경계값 (입력하지 않으면 기본값):"), sg.Input()],
        [sg.Button("확인")]
    ]
    window2 = sg.Window("폐사체 검출", layout2)

    while True:
        event2, values2 = window2.read()

        if event2 == sg.WIN_CLOSED:
            break
        elif event2 == "확인":
            read_path = values2[0]
            thresh = cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU if not values2[1] else int(values2[1])
            window2.close()
            remove_white(read_path, thresh, 1)

    window2.close()

def calculate_white_percent(image, thresh):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 0, 255, thresh)
    white_pixels = np.sum(mask == 0)
    total_pixels = np.prod(mask.shape)
    percent = (white_pixels / total_pixels) * 100
    return percent

def save_results_to_txt(results, save_path):
    with open(save_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.2f}%\n")

def calculate_chicken_size_window():
    layout = [
        [sg.Text("닭 크기 설정")],
        [sg.Text("1/4:"), sg.Input(key="-FOLDER_PATH1-"), sg.FolderBrowse()],
        [sg.Text("2/4:"), sg.Input(key="-FOLDER_PATH2-"), sg.FolderBrowse()],
        [sg.Text("3/4:"), sg.Input(key="-FOLDER_PATH3-"), sg.FolderBrowse()],
        [sg.Text("4/4:"), sg.Input(key="-FOLDER_PATH4-"), sg.FolderBrowse()],
        [sg.Button("닭 크기 저장")]
    ]

    window = sg.Window("닭 크기 설정", layout)

    results = {}  # Dictionary to store results

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == "닭 크기 저장":
            for i in range(1, 5):
                folder_path = values[f"-FOLDER_PATH{i}-"]
                img_files = [file for file in os.listdir(folder_path) if file.lower().endswith(".png")]

                total_percent = 0
                for img_file in img_files:
                    img_path = os.path.join(folder_path, img_file)
                    image = cv2.imread(img_path)
                    percent = calculate_white_percent(image, 9)  # Using 9 as base threshold
                    total_percent += percent

                avg_percent = total_percent / len(img_files)
                results[f"{i}/4"] = avg_percent

            save_layout = [
                [sg.Text("결과를 저장할 경로와 파일명을 선택하세요.")],
                [sg.Input(key="-SAVE_PATH-"), sg.FileSaveAs(), sg.Button("저장")]
            ]
            save_window = sg.Window("닭 크기 저장", save_layout)

            while True:
                save_event, save_values = save_window.read()
                if save_event == sg.WIN_CLOSED:
                    break
                elif save_event == "저장":
                    save_name = save_values["-SAVE_PATH-"]
                    if not save_name.lower().endswith(".txt"):
                        save_name += ".txt"

                    save_path = save_name
                    save_results_to_txt(results, save_path)  # Save results using custom function
                    save_window.close()
                    sg.popup("결과가 저장되었습니다.", title="저장 완료")
                    break  # Exit the loop after showing the popup

            save_window.close()
            window.close()

    window.close()


def calculate_chicken_size_from_file():
    layout = [
        [sg.Text("이미지 파일을 선택하세요:"), sg.Input(key="image_path"), sg.FileBrowse(file_types=(("Image Files", "*.png"),)), sg.Button("확인")]
    ]

    window = sg.Window("이미지 선택", layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == "확인":
            image_path = values["image_path"]
            calculate_and_show_chicken_size(image_path)
            break

    window.close()


def calculate_and_show_chicken_size(image_path):
    layout = [
        [sg.Text("닭 크기 정보를 담고 있는 txt 파일을 선택하세요:")],
        [sg.Input(key="txt_path"), sg.FileBrowse(file_types=(("Text Files", "*.txt"),))],
        [sg.Button("확인")]
    ]

    window = sg.Window("txt 파일 선택", layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == "확인":
            txt_path = values["txt_path"]
            image = cv2.imread(image_path)
            percent = calculate_white_percent(image, 9)  # 기본 임계값 9 사용
            size_info = find_nearest_chicken_size(percent, txt_path)
            sg.popup(f'닭 크기는 {size_info}입니다.')


    window.close()

def find_nearest_chicken_size(percent, txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    closest_size = None
    min_diff = float('inf')

    for line in lines:
        size, size_percent = line.split(':')
        size_percent = float(size_percent.strip()[:-1])  # '%' 제거 후 부동소수점으로 변환
        diff = abs(size_percent - percent)

        if diff < min_diff:
            min_diff = diff
            closest_size = size
    return closest_size

def create_main_window():
    sg.theme('DarkBlue')

    layout = [
        [sg.Text("Chicken_Gui", font=('Helvetica', 20))],
        [sg.Button("img 추출"), sg.Button("폐사체 검출"), sg.Button("닭 크기 설정"), sg.Button("닭 크기 찾기")]
    ]

    window = sg.Window("Chicken_Gui", layout, size=(400, 100))

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break
        elif event == "img 추출":
            window.hide()
            show_capture_frames_window()
            window.un_hide()
        elif event == "폐사체 검출":
            window.hide()
            show_remove_white_window()
            window.un_hide()
        elif event == "닭 크기 설정":
            window.hide()
            calculate_chicken_size_window()
            window.un_hide()
        elif event == "닭 크기 찾기":
            window.hide()
            calculate_chicken_size_from_file()
            window.un_hide()

    window.close()

def main():
    create_main_window()

if __name__ == "__main__":
    main()
