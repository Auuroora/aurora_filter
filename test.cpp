#include "define.h"
#include "header.h"

using namespace cv;
using namespace std;

/*********************************************************************
*	Mouse Callback Function
*********************************************************************/
void mouse_callback(int event, int x, int y, int flags, void *userdata)
{
	Mat *img = static_cast<Mat *>(userdata);

	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		imshow(TEST_WINDOW, imginfo.downsized_img);
		break;

	case EVENT_LBUTTONUP:
		imshow(TEST_WINDOW, imginfo.res_img);
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
void on_change_hue(int cur_pos, void *ptr)
{
	//update_hue(cur_pos - TRACKBAR_MID);
	//imshow(TEST_WINDOW, imginfo.res_img);
}

void on_change_saturation(int cur_pos, void *ptr)
{
	update_saturation(cur_pos - TRACKBAR_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.res_img);
}

void on_change_value(int cur_pos, void *ptr)
{
	update_value(cur_pos - TRACKBAR_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.res_img);
}

void on_change_temperature(int cur_pos, void *ptr)
{

	update_temperature(cur_pos - TRACKBAR_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.res_img);
}

void on_change_vibrance(int cur_pos, void *ptr)
{
	update_vibrance(cur_pos - 30);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.res_img);
}

void on_change_color_filter(int cur_pos, void *ptr)
{
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
	//add(imginfo.originImg, res, imginfo.res_img);
	//imshow(TEST_WINDOW, imginfo.res_img);
}

void on_change_highlight(int cur_pos, void *ptr)
{
	//imginfo.trackbar.splittone.highlight = cur_pos;
	//updateHighlightHue();
	//merge(imginfo.filterHsvSplit, 3, imginfo.hsvImg);
	//cvtColor(imginfo.hsvImg, imginfo.res_img, COLOR_HSV2BGR);
	//imshow(TEST_WINDOW, imginfo.res_img);
}