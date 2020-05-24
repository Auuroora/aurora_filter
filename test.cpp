#include "define.h"
#include "header.h"
// test
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
		imshow(TEST_WINDOW, imginfo.image.downsized);
		break;

	case EVENT_LBUTTONUP:
		imshow(TEST_WINDOW, imginfo.image.res);
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
	//imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_saturation(int cur_pos, void *ptr)
{
	update_saturation(cur_pos - TRACKBAR_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

// void on_change_value(int cur_pos, void *ptr)
// {
// 	update_value(cur_pos - TRACKBAR_MID);
// 	apply_filter();
// 	imshow(TEST_WINDOW, imginfo.image.res);
// }

void on_change_temperature(int cur_pos, void *ptr)
{

	update_temperature(cur_pos - TRACKBAR_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_vibrance(int cur_pos, void *ptr)
{
	update_vibrance(cur_pos - 30);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_highlight_hue(int cur_pos, void *ptr)
{
	//imginfo.trackbar.splittone.highlight = cur_pos;
	//updateHighlightHue();
	//merge(imginfo.filterHsvSplit, 3, imginfo.hsvImg);
	//cvtColor(imginfo.hsvImg, imginfo.image.res, COLOR_HSV2BGR);
	//imshow(TEST_WINDOW, imginfo.image.res);
}
void on_change_tint(int cur_pos, void *ptr)
{
	update_tint(cur_pos - TINT_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_grain(int cur_pos, void *ptr)
{
	update_grain(cur_pos - GRAIN_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_clarity(int cur_pos, void *ptr)
{
	update_clarity(cur_pos - CLARITY_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

int bright = 100;
int constrast = 100;

void on_change_bright(int cur_pos, void *ptr)
{
	bright = (cur_pos - BRIGHTNESS_MID);
	update_brightness_and_constrast(bright, constrast);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_constrast(int cur_pos, void *ptr)
{
	constrast = (cur_pos - CONSTRAST_MID);
	update_brightness_and_constrast(bright, constrast);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_exposure(int cur_pos, void *ptr)
{
	update_exposure(cur_pos - EXPOSURE_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_gamma(int cur_pos, void *ptr)
{
	update_gamma(cur_pos - GAMMA_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_vignette(int cur_pos, void *ptr)
{
	update_vignette(cur_pos - VIGNETTE_MID);
	apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}