#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;

// webcam을 이용해서 QR 코드 decoder 하는 방식
// openCV 가 설치되어있어야한다. 내기억으로는 이 컴퓨터에 Cpp 이라서 라이브러리를 빌드하고 가져갔던것 같다. 



int main() {
    cv::VideoCapture cap(0); 
    

    if (!cap.isOpened()) {
        std::cerr << "카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    int width = 640;  
    int height = 480; 
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    cv::QRCodeDetector qrDecoder;   
    cv::QRCodeDetector detector;


   
    while (true) {
        cv::Mat frame, gray;
        cap >> frame; // 카메라로부터 새로운 프레임을 가져옴

        // 프레임이 비어있는지 확인
        if (frame.empty()) {
            std::cerr << "빈 프레임 받음!" << std::endl;
            break;
        }

    

        //QR 코드는 gray 에서 가져가는 경우가 많다 -> RGB 보다 픽셀 데이터양이 적기 떄문에 더 빠르다.
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<cv::Point> points;

        //그냥 gray 이미지 
        bool detected = qrDecoder.detect(gray, points);
        if(detected){
            cout << "QR code Detection 1" << endl;
            polylines(frame, points, true, Scalar(0, 255, 255), 4);
        }



        //아래 데이터는 그냥 gray 이미지를 더 효율적으로 확인하기 위해서 opencv 의 여러 이미지 프로세싱을 하는것 -> 실제 테스트상으로는 아예 안한것 보단 낫지만 어떤 조합이 나은지는 좀더 찾아봐야함

        // 명암 대비 조정
        cv::Mat contrastAdjusted;
        convertScaleAbs(gray, contrastAdjusted, 1.5, 0); // 대비를 1.5배 증가

        detected = qrDecoder.detect(contrastAdjusted, points);
        if(detected){
            cout << "QR code Detection 2" << endl;
            polylines(frame, points, true, Scalar(255, 0, 255), 3);
        }

        //Adaptive threshold 기법 
        cv::Mat adaptiveImage;
        adaptiveThreshold(gray, adaptiveImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);

        detected = qrDecoder.detect(adaptiveImage, points);
        if(detected){
            cout << "QR code Detection 3" << endl;
            polylines(frame, points, true, Scalar(255, 255, 0), 2);
        }


        // 히스토그램 평활화
        Mat equalized;
        equalizeHist(gray, equalized);
        detected = qrDecoder.detect(equalized, points);
        if(detected){
            cout << "QR code Detection 4" << endl;
            polylines(frame, points, true, Scalar(0, 255, 0), 2);
        }


        // 샤프닝
        Mat sharpened;
        GaussianBlur(gray, sharpened, Size(0, 0), 3);
        addWeighted(gray, 1.5, sharpened, -0.5, 0, sharpened);
        detected = qrDecoder.detect(sharpened, points);
        if(detected){
            cout << "QR code Detection 5" << endl;
            polylines(frame, points, true, Scalar(255, 0, 0), 2);
        }


        // 이진화
        Mat binarized;
        threshold(gray, binarized, 0, 255, THRESH_BINARY | THRESH_OTSU);
        detected = qrDecoder.detect(binarized, points);
        if(detected){
            cout << "QR code Detection 6" << endl;
            polylines(frame, points, true, Scalar(0, 0, 255), 2);
        }








        // 화면에 프레임 표시
        cv::imshow("카메라", frame);

        // ESC 키를 누르면 루프 종료
        if (cv::waitKey(1) == 27) {
            break;
        }
    }
    // 사용이 끝난 후 카메라 연결 해제
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
