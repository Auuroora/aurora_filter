#include "define.h"
#include "header.h"

/*********************************************************************
*	Mouse Callback Function
*********************************************************************/
void mouse_callback(int event, int x, int y, int flags, void *userdata) {
	cv::Mat* img = static_cast<cv::Mat*>(userdata);
	cv::Mat res;

	switch (event) {
	case cv::EVENT_LBUTTONDOWN:
		cv::imshow(TEST_WINDOW, get_preview_image(*img, imginfo.image.logo));
		break;

	case cv::EVENT_LBUTTONUP:
		cv::imshow(TEST_WINDOW, *img);
		break;

	case cv::EVENT_MOUSEMOVE:
		break;
	}
}

/*********************************************************************
*	Trackbar Callback Function
*********************************************************************/
void on_change_hue(int curPos, void* ptr) {
	imginfo.update_hue(curPos - TRACKBAR_MID);
	imginfo.apply_filter();
	cv::imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_saturation(int curPos, void* ptr) {
	imginfo.update_saturation(curPos - TRACKBAR_MID);
	imginfo.apply_filter();
	cv::imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_lightness(int curPos, void* ptr) {
	imginfo.update_lightness(curPos - TRACKBAR_MID);
	imginfo.apply_filter();
	cv::imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_temperature(int curPos, void* ptr) {
	imginfo.update_temperature(curPos - TRACKBAR_MID);
	imginfo.apply_filter();
	cv::imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_vibrance(int curPos, void* ptr) {
	imginfo.update_vibrance(curPos - TRACKBAR_MID);
	imginfo.apply_filter();
	cv::imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_highlight_hue(int curPos, void* ptr) {
	imginfo.update_highlight_hue(curPos - TRACKBAR_MID);
	imginfo.apply_filter();
	cv::imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_highlight_saturation(int curPos, void* ptr) {
	imginfo.update_highlight_saturation(curPos - TRACKBAR_MID);
	imginfo.apply_filter();
	cv::imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_shadow_hue(int curPos, void* ptr) {
	imginfo.update_shadow_hue(curPos - TRACKBAR_MID);
	imginfo.apply_filter();
	cv::imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_shadow_saturation(int curPos, void* ptr) {
	imginfo.update_shadow_saturation(curPos - TRACKBAR_MID);
	imginfo.apply_filter();
	cv::imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_tint(int cur_pos, void *ptr) {
	imginfo.update_tint(cur_pos - TINT_MID);
	imginfo.apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_grain(int cur_pos, void *ptr) {
	imginfo.update_grain(cur_pos - GRAIN_MID);
	imginfo.apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_clarity(int cur_pos, void *ptr) {
	imginfo.update_clarity(cur_pos - CLARITY_MID);
	imginfo.apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

int bright = 100;
int constrast = 100;

void on_change_bright(int cur_pos, void *ptr) {
	bright = (cur_pos - BRIGHTNESS_MID);
	imginfo.update_brightness_and_constrast(bright, constrast);
	imginfo.apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_constrast(int cur_pos, void *ptr) {
	constrast = (cur_pos - CONSTRAST_MID);
	imginfo.update_brightness_and_constrast(bright, constrast);
	imginfo.apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_exposure(int cur_pos, void *ptr) {
	imginfo.update_exposure(cur_pos - EXPOSURE_MID);
	imginfo.apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_gamma(int cur_pos, void *ptr) {
	imginfo.update_gamma(cur_pos - GAMMA_MID);
	imginfo.apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}

void on_change_vignette(int cur_pos, void *ptr) {
	imginfo.update_vignette(cur_pos - VIGNETTE_MID);
	imginfo.apply_filter();
	imshow(TEST_WINDOW, imginfo.image.res);
}