import os
import cv2
import PySimpleGUI as sg
import numpy as np
import random
import matplotlib.pyplot as plt

lower_color = np.array([104, 128, 98])
upper_color = np.array([248, 247, 236])

# lower_color = np.array([0, 124, 26])
# upper_color = np.array([230, 255, 255])

def is_valid_path(path):
    if os.path.exists(path) and path is not None:
        return True
    else:
        return False

def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def find_objects(image, distance_threshold):
    # 이미지에서 좌표 찾기
    mask = cv2.inRange(image, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    random.shuffle(contours[0])

    # 좌표들을 저장할 리스트
    coordinates = []

    # 좌표 추출
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
        coordinates.append((cx, cy))

    random.shuffle(coordinates)

    # 중복된 좌표 제거
    filtered_coordinates = []
    for i in range(len(coordinates)):
        cx1, cy1 = coordinates[i]
        is_duplicate = False
        for j in range(i + 1, len(coordinates)):
            cx2, cy2 = coordinates[j]
            if abs(cx2 - cx1) <= distance_threshold and abs(cy2 - cy1) <= distance_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_coordinates.append((cx1, cy1))
    return filtered_coordinates

def find_and_draw_objects(image_path, distance_threshold):
    # 이미지 로드 및 처리
    image = cv2.imread(image_path)

    filtered_coordinates = find_objects(image, distance_threshold)

    # 이미지에 동그라미 그리기
    image_with_circles = image.copy()
    for coord in filtered_coordinates:
        cv2.circle(image_with_circles, coord, 10, (0, 0, 255), 2)

    return image_with_circles, filtered_coordinates

def show_result_window(image, coordinates):
    resized_image = resize_image(image, 50)  # 이미지 크기 25%로 조절

    layout = [
        [sg.Image(data=cv2.imencode('.png', resized_image)[1].tobytes())],
        [sg.Button("저장"), sg.Button("닫기")]
    ]

    window = sg.Window("결과 확인", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "닫기":
            break
        elif event == "저장":
            write_path = sg.popup_get_file("저장 경로 선택", save_as=True, default_extension=".jpg", file_types=(("JPEG Files", "*.jpg"),))
            if is_valid_path(write_path):
                cv2.imwrite(write_path, image)

                # 좌표 텍스트 파일도 저장
                coord_output_file = write_path.replace(".jpg", "_coordinates.txt")
                with open(coord_output_file, 'w') as f:
                    for coord in coordinates:
                        f.write(f"{coord[0]},{coord[1]}\n")

                sg.popup("이미지와 좌표가 저장되었습니다.")
            else:
                sg.popup("저장 경로를 알맞게 설정하세요!")

    window.close()

def find_and_draw_objects_window():
    layout = [
        [sg.Text("객체 찾기")],
        [sg.Text("이미지 선택:"), sg.Input(), sg.FileBrowse(file_types=(("PNG Files", "*.png"),))],
        [sg.Text("닭 사이 thresh 경계값 (입력하지 않으면 기본값 = 20):"), sg.Input()],
        [sg.Button("확인")]
    ]

    window = sg.Window("객체 찾기", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == "확인":
            img_path = os.path.normpath(values[0])

            if is_valid_path(img_path):
                distance_threshold = 20 if not values[1] else int(values[1])
                image, coordinates = find_and_draw_objects(img_path, distance_threshold)
                window.close()
                show_result_window(image, coordinates)
            else:
                sg.popup("이미지 경로를 알맞게 설정하세요!")

    window.close()

def find_coordinates_and_convert2xml(read_path, write_path, distance_threshold):
    # 모든 png 파일에 대해서 실행
    for filename in os.listdir(read_path):
        if filename.endswith('.png'):
            # _gt 이전 부분을 제외한 XML 파일 이름 생성
            xml_filename = filename.split('_gt')[0] + '.xml'
            xml_filepath = os.path.join(write_path, xml_filename)

            # 이미지 로드 및 처리
            image = cv2.imread(os.path.join(read_path, filename))

            filtered_coordinates = find_objects(image, distance_threshold)

            # XML 파일 생성 및 작성
            with open(xml_filepath, 'w') as out_f:
                out_f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                out_f.write('<annotation>\n')
                out_f.write(f'\t<folder>{os.path.basename(write_path)}</folder>\n')
                out_f.write(f'\t<filename>{xml_filename[:-4]}.png</filename>\n')
                out_f.write(f'\t<path>{xml_filepath}</path>\n')
                # 나머지 XML 작성 코드
                for coord in filtered_coordinates:
                    xmin = max(coord[0] - 5, 0)
                    ymin = max(coord[1] - 5, 0)
                    xmax = coord[0] + 5
                    ymax = coord[1] + 5
                    out_f.write('\t<object>\n')
                    out_f.write('\t\t<name>color</name>\n')
                    out_f.write('\t\t<pose>Unspecified</pose>\n')
                    out_f.write('\t\t<truncated>0</truncated>\n')
                    out_f.write('\t\t<difficult>0</difficult>\n')
                    out_f.write('\t\t<bndbox>\n')
                    out_f.write(f'\t\t\t<xmin>{xmin}</xmin>\n')
                    out_f.write(f'\t\t\t<ymin>{ymin}</ymin>\n')
                    out_f.write(f'\t\t\t<xmax>{xmax}</xmax>\n')
                    out_f.write(f'\t\t\t<ymax>{ymax}</ymax>\n')
                    out_f.write('\t\t</bndbox>\n')
                    out_f.write('\t</object>\n')
                out_f.write('</annotation>\n')
            print(f"{filename} {xml_filename}으로 변환 완료")

    sg.popup("모든 파일 변환 완료")

def find_coordinates_and_convert2xml_window():
    layout = [
        [sg.Text("좌표 -> XML 변환")],
        [sg.Text("read_path:"), sg.Input(), sg.FolderBrowse()],
        [sg.Text("write_path:"), sg.Input(), sg.FolderBrowse()],
        [sg.Text("닭 사이 thresh 경계값 (입력하지 않으면 기본값 = 20):"), sg.Input()],
        [sg.Button("확인")]
    ]

    window = sg.Window("좌표 -> XML 변환", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == "확인":
            read_path = os.path.normpath(values[0])  # 읽을 디렉토리 경로
            write_path = os.path.normpath(values[1])  # 저장할 디렉토리 경로

            if is_valid_path(read_path) and is_valid_path(write_path):
                distance_threshold = 20 if not values[2] else int(values[2])
                find_coordinates_and_convert2xml(read_path, write_path, distance_threshold)
            else:
                sg.popup("read_path와 write_path를 알맞게 입력하세요!")

    window.close()

def find_ratio_distribution_window():
    layout = [
        [sg.Text("분포 찾기")],
        [sg.Button("기본 분포 찾기"), sg.Button("사용자 지정 분포 찾기")]
    ]

    window = sg.Window("분포 찾기", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == "닫기":
            break
        elif event == "기본 분포 찾기":
            find_default_ratio_distribution_window()
        elif event == "사용자 지정 분포 찾기":
            find_custom_ratio_distribution_window()

    window.close()

def find_custom_ratio_distribution_window():
    layout = [
        [sg.Text("사용자 지정 분포 찾기")],
        [sg.Text("학습 결과 선택:"), sg.Input(), sg.FileBrowse()],
        [sg.Text("원본 이미지 선택:"), sg.Input(), sg.FileBrowse()],
        [sg.Text("닭 사이 thresh 경계값 (입력하지 않으면 기본값 = 20):"), sg.Input()],
        [sg.Button("확인")]
    ]

    window = sg.Window("사용자 지정 분포 찾기", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == "확인":
            result_image_path = os.path.normpath(values[0])  # 학습 결과 이미지 경로
            original_image_path = os.path.normpath(values[1])  # 원본 이미지 경로

            if is_valid_path(result_image_path) and is_valid_path(original_image_path):
                distance_threshold = 20 if not values[2] else int(values[2])
                find_and_display_custom_ratio(result_image_path, original_image_path, distance_threshold)
                window.close()
            else:
                sg.popup("학습 결과 이미지 경로와 원본 이미지 경로를 알맞게 입력하세요!")

    window.close()

def find_and_display_custom_ratio(result_image_path, original_image_path, distance_threshold):
    result_image = cv2.imread(result_image_path)
    original_image = cv2.imread(original_image_path)

    filtered_coordinates = find_objects(result_image, distance_threshold)

    image_with_circles = original_image.copy()
    for coord in filtered_coordinates:
        cv2.circle(image_with_circles, coord, 10, (255, 0, 0), 2)

    selected_coordinates = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_coordinates

        if event == cv2.EVENT_LBUTTONDOWN:
            selected_coordinates = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            selected_coordinates.append((x, y))
            image_with_boxes = calculate_and_display_custom_ratio(image_with_circles, filtered_coordinates, selected_coordinates)
            cv2.imshow("Image with Boxes", resize_image(image_with_boxes, 50))

    cv2.imshow("Image with Circles", resize_image(image_with_circles, 50))
    cv2.setMouseCallback("Image with Circles", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("Image with Circles", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def calculate_and_display_custom_ratio(image, coordinates, selected_coordinates):
    image_with_boxes = image.copy()

    x1, y1 = selected_coordinates[0]
    x2, y2 = selected_coordinates[1]

    selected_coordinates_original = (x1 * 2, y1 * 2, x2 * 2, y2 * 2)

    cropped_filtered_coordinates = []
    for coord in coordinates:
        cx, cy = coord
        if selected_coordinates_original[0] <= cx < selected_coordinates_original[2] and \
                selected_coordinates_original[1] <= cy < selected_coordinates_original[3]:
            cropped_filtered_coordinates.append((cx, cy))

    ratio, ratio_text = calculate_custom_ratio(cropped_filtered_coordinates, len(coordinates))

    cv2.rectangle(image_with_boxes, (selected_coordinates_original[0], selected_coordinates_original[1]),
                  (selected_coordinates_original[2], selected_coordinates_original[3]), (0, 255, 0), 2)

    text = f"Custom Ratio: {ratio:.4f} ({ratio_text})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.putText(image_with_boxes, text,
                (selected_coordinates_original[0], selected_coordinates_original[1] + text_size[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return image_with_boxes


def calculate_custom_ratio(coordinates, total_object_count):
    region_coordinates = []
    for coord in coordinates:
        cx, cy = coord
        region_coordinates.append((cx, cy))

    region_object_count = len(region_coordinates)
    ratio = region_object_count / total_object_count

    return ratio, f"{region_object_count}/{total_object_count}"

def calculate_ratios(image, coordinates, num_horizontal_parts, num_vertical_parts):
    height, width, _ = image.shape
    horizontal_step = width // num_horizontal_parts
    vertical_step = height // num_vertical_parts

    ratios = []
    for i in range(num_vertical_parts):
        for j in range(num_horizontal_parts):
            xmin = j * horizontal_step
            ymin = i * vertical_step
            xmax = xmin + horizontal_step
            ymax = ymin + vertical_step
            region_coordinates = []
            for coord in coordinates:
                cx, cy = coord
                if xmin <= cx < xmax and ymin <= cy < ymax:
                    region_coordinates.append(coord)
            region_object_count = len(region_coordinates)
            ratio = region_object_count / len(coordinates)
            ratios.append([ratio, f"{region_object_count}/{len(coordinates)}"])
            print(ratio, f"{region_object_count}/{len(coordinates)}")
    return ratios


def find_default_ratio_distribution(result_image_path, original_image_path):
    layout = [
        [sg.Text("기본 분포 찾기 설정")],
        [sg.Text("닭 사이 thresh 경계값 (입력하지 않으면 기본값 = 20):"), sg.Input()],
        [sg.Text("가로로 나눌 수 (입력하지 않으면 기본값 = 10):"), sg.Input()],
        [sg.Text("세로로 나눌 수 (입력하지 않으면 기본값 = 5):"), sg.Input()],
        [sg.Button("확인")]
    ]

    window = sg.Window("기본 분포 찾기 설정", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == "확인":
            result_image = cv2.imread(result_image_path)
            original_image = cv2.imread(original_image_path)

            distance_threshold = 20 if not values[0] else int(values[0])
            num_horizontal_parts = 10 if not values[1] else int(values[1])
            num_vertical_parts = 5 if not values[2] else int(values[2])

            filtered_coordinates = find_objects(result_image, distance_threshold)
            ratios = calculate_ratios(result_image, filtered_coordinates, num_horizontal_parts, num_vertical_parts)

            image_with_circles = original_image.copy()
            for coord in filtered_coordinates:
                cv2.circle(image_with_circles, coord, 10, (255, 0, 0), 2)

            height, width, _ = original_image.shape
            horizontal_step = width // num_horizontal_parts
            vertical_step = height // num_vertical_parts

            for i in range(num_vertical_parts):
                for j in range(num_horizontal_parts):
                    x = j * horizontal_step
                    y = i * vertical_step
                    cv2.rectangle(image_with_circles, (x, y), (x + horizontal_step, y + vertical_step), (0, 0, 255), 3)
                    ratio = ratios[i * num_horizontal_parts + j][0]
                    cv2.putText(image_with_circles, f"{ratio:.4f}, {ratios[i * num_horizontal_parts + j][1]}", (x + 10, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)

            resized_image = resize_image(image_with_circles, 50)
            cv2.imshow("Image with Circles and Ratios", resized_image)
            cv2.imwrite("asdf.png", image_with_circles)
            cv2.waitKey(0)

        else:
            sg.popup("학습 결과 이미지 경로와 원본 이미지 경로를 알맞게 입력하세요!")

    window.close()

def find_default_ratio_distribution_window():
    layout = [
        [sg.Text("기본 분포 찾기")],
        [sg.Text("학습 결과 선택:"), sg.Input(), sg.FileBrowse()],
        [sg.Text("원본 이미지 선택:"), sg.Input(), sg.FileBrowse()],
        [sg.Button("확인")]
    ]

    window = sg.Window("기본 분포 찾기", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == "확인":
            result_image_path = os.path.normpath(values[0])  # 학습 결과 이미지 경로
            original_image_path = os.path.normpath(values[1])  # 원본 이미지 경로

            if is_valid_path(result_image_path) and is_valid_path(original_image_path):
                window.un_hide()
                find_default_ratio_distribution(result_image_path, original_image_path)
                window.close()
            else:
                sg.popup("학습 결과 이미지 경로와 원본 이미지 경로를 알맞게 입력하세요!")

    window.close()

def create_heatmap_window():
    layout = [
        [sg.Text("히트맵 생성")],
        [sg.Text("학습 결과 선택:"), sg.Input(), sg.FileBrowse()],
        [sg.Text("원본 이미지 선택:"), sg.Input(), sg.FileBrowse()],
        [sg.Text("히트맵 정밀도 선택:"), sg.Input()],
        [sg.Button("확인")]
    ]

    window = sg.Window("히트맵 생성", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == "확인":
            result_image_path = os.path.normpath(values[0])  # 학습 결과 이미지 경로
            original_image_path = os.path.normpath(values[1])  # 원본 이미지 경로
            square_size = 100 if not values[2] else int(values[2])

            if is_valid_path(result_image_path) and is_valid_path(original_image_path):
                window.close()
                create_and_overlay_heatmap(result_image_path, original_image_path, square_size)
            else:
                sg.popup("학습 결과 이미지 경로와 원본 이미지 경로를 알맞게 입력하세요!")

    window.close()

def create_and_overlay_heatmap(result_image_path, original_image_path, square_size):
    result_image = cv2.imread(result_image_path)
    filtered_coordinates = find_objects(result_image, distance_threshold=20)

    original_image = cv2.imread(original_image_path)

    heatmap_data = np.zeros((original_image.shape[0], original_image.shape[1]))

    for y in range(0, original_image.shape[0] - square_size + 1, 5):
        for x in range(0, original_image.shape[1] - square_size + 1, 5):
            square_counter = 0  # 사각형 내의 객체 개수를 세는 카운터
            for coord in filtered_coordinates:
                if x <= coord[0] < x + square_size and y <= coord[1] < y + square_size:
                    square_counter += 1

            heatmap_data[y:y + square_size, x:x + square_size] += square_counter

    heatmap_data = heatmap_data / np.max(heatmap_data)

    # 히트맵 그리기
    plt.matshow(heatmap_data, cmap=plt.cm.Spectral, origin='upper',
                extent=[0, original_image.shape[1], original_image.shape[0], 0])

    # 원본 이미지 위에 히트맵 덮어 씌우기
    heatmap_overlay = plt.gca().get_images()[0]
    heatmap_data = heatmap_overlay.get_array()  # 2차원 배열로 히트맵 데이터 얻기

    heatmap_image = plt.cm.Spectral(heatmap_data)[:, :, :3]  # Using Spectral colormap for color mapping
    heatmap_image = (heatmap_image * 255).astype(np.uint8)

    heatmap_overlay = cv2.addWeighted(original_image, 0.3, heatmap_image, 0.7, 0)

    cv2.imshow("Heatmap with Original Image", resize_image(heatmap_overlay, 50))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def create_main_window():
    sg.theme('DarkBlue')

    layout = [
        [sg.Text("SAFECount_GUI", font=('Helvetica', 20))],
        [sg.Button("객체 찾기"), sg.Button("좌표 -> xml"), sg.Button("분포 찾기"), sg.Button("히트맵 생성")]
    ]

    window = sg.Window("SAFECount GUI", layout)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        elif event == "객체 찾기":
            window.un_hide()
            find_and_draw_objects_window()
        elif event == "좌표 -> xml":
            window.un_hide()
            find_coordinates_and_convert2xml_window()
        elif event == "분포 찾기":
            window.un_hide()
            find_ratio_distribution_window()
        elif event == "히트맵 생성":
            window.un_hide()
            create_heatmap_window()

    window.close()

def main():
    create_main_window()

if __name__ == "__main__":
    main()
