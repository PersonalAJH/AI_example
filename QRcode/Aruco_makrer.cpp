/*
    arUco marker 는 컴퓨터 비전 분야에서 많이 사용하며 간단하면서도 강력한 마커기반 추적 시스템입니다.
    이 마커들은 검은색과 흰색의 사각형 패턴으로 구성되어 있으며, 각각의 패턴은 유일한 식별자(ID)를 가집니다. 
    ArUco 라이브러리는 이러한 마커들을 실시간으로 감지하고, 마커의 위치와 방향을 추정할 수 있는 기능을 제공합니다.

    arUco maker 에는 여러가지 특징을 가진다. 
    1. 각 ArUco 마커에는 고유한 식별자가 있어서 여러 마커를 동시에 사용ㅎ랄 수 있으며 각각을 구별할 수 있다.
    2. 실시간 시스템에서 빠르게 감지하고 추적할 수 있으며 낮은 해상도의 이미지에서도 잘 작동한다.
    3. 다양한 크기와 구성으로 마커를 생성할 수 있으며 카메 캘리브레이션과 같은 추가 정보를 통해 3D 위치와 방향추정의 정확도도 높은편
    4. 오픈소스이며 OpenCV에서 많은 툴을 제공함

    QR 코드와 비교하여 더 높은 신뢰도로 더 빠르게 인식할 수 있으며 낮은 해상도에서 잘 작동한다.
    또한 높은신뢰도를 가지기 떄문에 방향을 확인하는 (Orientation) 것 또한 유리함

    단점은 QR코드는 여러 정보들을 안에 넣을 수 있지만 이건 마커의 고유ID밖에는 불가능

*/


#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

void detectArucoAndEstimatePose(cv::VideoCapture& cap, const cv::Ptr<cv::aruco::Dictionary>& dictionary) {
    cv::Mat frame, imageCopy;
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
    cv::aruco::DetectorParameters parameters;
    cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();

    // 카메라 매트릭스와 왜곡 계수 - 실제 측정값으로 대체해야 합니다.
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frame.copyTo(imageCopy);

        // ArUco 마커 감지
        // dictionary : ArUco 마커의 사전을 나타내며, 감지하려는 마커의 종류를 정의합니다. OpenCV의 ArUco 모듈은 여러 사전을 제공하며, 각 사전은 다른 마커 ID 집합을 가지고 있습니다 -> 4X4, 5X5, 6X6 이렇게 선택된 것만 지정됨 
        // markerConers : 마커의 꼭지점들 집합? (x,y)
        // markerIds : 마커 아이디 
        // detectorParams : cv::aruco::DetectorParameters 타입의 객체입니다. 이 매개변수를 통해 감지 알고리즘의 세부 동작을 조정할 수 있으며, 대표적인 설정으로는 코너 검출의 적응형 임계값, 모서리 정제 단계의 활성화 여부 등이 있습니다.
        cv::aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, detectorParams, rejectedCandidates);

        if (!markerIds.empty()) {
            cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);

            std::vector<cv::Vec3d> rvecs, tvecs;

            // 마커의 포즈 추정
            cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

            // 마커가 여러개 확인 됐을 경우도 있음
            for (int i = 0; i < markerIds.size(); i++) {
                cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.03);

                // Orientation 벡터 계산 (0,0,1 벡터 변환) -> 마커가 튀어나오는 쪽에 대한 게산식 
                cv::Mat rotationMatrix;
                cv::Rodrigues(rvecs[i], rotationMatrix);        // 로드리게스라고 3개의 아웃풋으로 하나의 축과 회전각을 나타내는 변수임
                cv::Mat forwardVector = (cv::Mat_<double>(3,1) << 0, 0, 1); // Z축 방향
                cv::Mat transformedVector = rotationMatrix * forwardVector; // 변환된 방향 벡터

                std::cout << "Marker ID: " << markerIds[i] << " Orientation Vector: " << transformedVector.t() << std::endl;
            }
        }

        cv::imshow("Detected ArUco markers", imageCopy);
        char key = (char)cv::waitKey(10);
        if (key == 27) break;
    }
}

int main() {
    cv::VideoCapture inputVideo;
    inputVideo.open(0); // 0은 첫 번째 카메라를 의미

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    detectArucoAndEstimatePose(inputVideo, dictionary);

    return 0;
}
