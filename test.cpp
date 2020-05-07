#include "define.h"
#include "header.h"

using namespace cv;
using namespace std;

/*********************************************************************
*	Mouse Callback Function
*********************************************************************/
void mouseCallback(int event, int x, int y, int flags, void *userdata) {
	Mat* img = static_cast<Mat*>(userdata);

	switch (event) {
	case EVENT_LBUTTONDOWN:
		imshow(TEST_WINDOW, imginfo.downsizedImg);
		break;

	case EVENT_LBUTTONUP:
		imshow(TEST_WINDOW, imginfo.resImg);
		break;

	case EVENT_MOUSEMOVE:
		//if (img->channels() == 1) {
		//	cout << "S: " << (int)img->at<Vec3b>(y, x)[0] << endl;
		//}
		//if (img->channels() == 3) {
		//	cout << "H: " << (int)img->at<Vec3b>(y, x)[0];
		//	cout << " S: " << (int)img->at<Vec3b>(y, x)[1];
		//	cout << " V: " << (int)img->at<Vec3b>(y, x)[2] << endl;
		//}
		break;
	}
}

/*********************************************************************
*	Trackbar Callback Function
*********************************************************************/
void onChangeHue(int curPos, void* ptr) {
	//updateHue(curPos - TRACKBAR_MID);
	//imshow(TEST_WINDOW, imginfo.resImg);
}

void onChangeSaturation(int curPos, void* ptr) {
	updateSaturation(curPos - TRACKBAR_MID);
	applyFilter();
	imshow(TEST_WINDOW, imginfo.resImg);
}

void onChangeValue(int curPos, void* ptr) {
	updateValue(curPos - TRACKBAR_MID);
	applyFilter();
	imshow(TEST_WINDOW, imginfo.resImg);
}

void onChangeTemperature(int curPos, void* ptr) {
	

	updateTemperature(curPos - TRACKBAR_MID);
	applyFilter();

	
	imshow(TEST_WINDOW, imginfo.resImg);
}

void onChangeVibrance(int curPos, void* ptr) {
	//imginfo.trackbar.color.vibrance = curPos - TRACKBAR_MID;
	//updateVibrance();
	//merge(imginfo.filterHsvSplit, 3, imginfo.hsvImg);
	//cvtColor(imginfo.hsvImg, imginfo.resImg, COLOR_HSV2BGR);
	//imshow(TEST_WINDOW, imginfo.resImg);
}

void onChangeColorFilter(int curPos, void* ptr) {
	//int r = imginfo.originImg.rows;
	//int c = imginfo.originImg.cols;
	//int redPos = getTrackbarPos("R", SET_WINDOW);
	//int greenPos = getTrackbarPos("G", SET_WINDOW);
	//int bluePos = getTrackbarPos("B", SET_WINDOW);
	//Mat v[] = {
	//	Mat::ones(r, c, CV_8U) * bluePos,
	//	Mat::ones(r, c, CV_8U) * greenPos,
	//	Mat::ones(r, c, CV_8U) * redPos
	//};
	//Mat res;
	//merge(v, 3, res);
	//add(imginfo.originImg, res, imginfo.resImg);
	//imshow(TEST_WINDOW, imginfo.resImg);
}

void onChangeHighlight(int curPos, void* ptr) {
	//imginfo.trackbar.splittone.highlight = curPos;
	//updateHighlightHue();
	//merge(imginfo.filterHsvSplit, 3, imginfo.hsvImg);
	//cvtColor(imginfo.hsvImg, imginfo.resImg, COLOR_HSV2BGR);
	//imshow(TEST_WINDOW, imginfo.resImg);
}