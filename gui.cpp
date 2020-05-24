#include "define.h"
#include "header.h"

using namespace cv;
using namespace std;

void mouseCallback(int event, int x, int y, int flags, void *userdata) {
	Mat* img = static_cast<Mat*>(userdata);

	switch (event) {
	case EVENT_LBUTTONDOWN:
		imshow(TEST_WINDOW, imginfo.originImg);
		break;

	case EVENT_LBUTTONUP:
		imshow(TEST_WINDOW, imginfo.resImg);
		break;
	case EVENT_MOUSEMOVE:
		if (img->channels() == 1) {
			cout << "H: " << (int)img->at<Vec3b>(y, x)[0] << endl;
		}
		if (img->channels() == 3) {
			cout << "H: " << (int)img->at<Vec3b>(y, x)[0];
			cout << " S: " << (int)img->at<Vec3b>(y, x)[1];
			cout << " V: " << (int)img->at<Vec3b>(y, x)[2] << endl;
		}
	}
}

void onChangeHue(int curPos, void* ptr) {
	updateHue_pararell();
	merge(imginfo.filterSplit, 3, imginfo.hsvImg);
	cvtColor(imginfo.hsvImg, imginfo.resImg, COLOR_HSV2BGR);
	imshow(TEST_WINDOW, imginfo.resImg);
}

void onChangeSaturation(int curPos, void* ptr) {

}

void onChangeValue(int curPos, void* ptr) {

}