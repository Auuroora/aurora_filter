#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core.hpp>
#include <iomanip>
#include <stdlib.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

// enum class BGRIndex;

// �� ���� �ε���
namespace BGR{
	typedef enum {
		B=0,
		G,
		R
	} BGRIndex;
}
namespace HSV{
	typedef enum{
		H=0,
		S,
		V
	} HSVIndex;
}

namespace HLS{
	typedef enum {
		H=0,
		L,
		S
	} HLSIndex;	
}

// enum class BGRIndex :int 
// {
// 	B = 0,
// 	G = 1,
// 	R = 2,
// };

// // �� ���� �ε���
// enum class HSVIndex :int
// {
// 	H = 0,
// 	S = 1,
// 	V = 2,
// };

// // �� ���� �ε���
// enum class HLSIndex :int
// {
// 	H = 0,
// 	L = 1,
// 	S = 2
// };

// �۾����� ��� ���� �� ���⿡
class WorkingImgInfo
{
public:
	WorkingImgInfo(){

	};

	int row; // �ٿ����¡ �� ���� ����
	int col; // �ٿ����¡ �� ���� ����
	int changed_color_space = 0;

	/***********************************************************************************************/
	struct Image
	{
		Mat downsized;			 // �ٿ����¡ �� �̹���
		Mat bgr, hls, hsv, res;	 // bgr�̹���, hsv�̹���, ���� �����
		vector<Mat> bgr_origins; // split�� ����(bgr)
		vector<Mat> hls_origins; // split�� ����(hls)
		vector<Mat> hsv_origins; // split�� ����(hsv)
		vector<Mat> res_split;
	} image;

	Mat downsized_img;	  // �ٿ����¡ �� �̹���
	Mat bgr_img, hsv_img; // bgr�̹���, hsv�̹���
	Mat res_img;		  // ���� �����

	vector<Mat> bgr_split; //bgrImg�� split�� ����
	vector<Mat> hsv_split; //hsvImg�� split�� ����
	vector<Mat> res_split;
	/***********************************************************************************************/

	// filter
	struct Filter
	{
		Mat diff;		   // ���� ������ ���� 1ä�� ���
		vector<Mat> diffs; //���� ������ ���� 3ä�� ���
		Mat bgr_filter;	   // bgr����ġ�� ��ϵǾ� �ִ� ����
		Mat hsv_filter;	   // hsv����ġ�� ��ϵǾ� �ִ� ����
		Mat hls_filter;	   // hsv����ġ�� ��ϵǾ� �ִ� ����

		Mat clarity_filter;
		Mat clarity_mask_U;
		Mat clarity_mask_S;
		vector<Mat> clarity_mask_split;

		Mat gaussian_kernel;

		Mat gamma_mask;

		Mat grain_mask;
		Mat salt_mask;
		Mat pepper_mask;

		Mat exposure_mask;

		vector<Mat> bgr_filters; // split�� ����(bgr)
		vector<Mat> hls_filters; // split�� ����(hls)
		vector<Mat> hsv_filters; // split�� ����(hsv)
	} filter;

	// �� ����� ����ġ ���
	struct Weight
	{
		Mat blue, green, red;
		Mat hue, sat, val;
	} weight;

	// trackbar pos
	// ���� Ʈ���� ���� ������ ������
	struct Trackbar
	{

		int temperature = 0;
		int hue;
		int saturation = 0;
		int lightness = 0;
		int vibrance = 0;
		int highlight_hue = 0;
		int highlight_sat = 0;

		int brightness = 0;
		int constrast = 0;
		int tint = 0;
		int clarity = 0;
		int exposure = 0;
		int gamma = 0;
		int grain = 0;
		int vignette = 0;

	} trackbar;

	// getter & setter
	Mat get_origin_img()
	{
		return this->origin_img;
	}

	void set_origin_img(Mat img)
	{
		this->origin_img = img.clone();
	}

	Mat get_res_img()
	{
		return this->res_img;
	}

private:
	Mat origin_img; // ���� �Ұ��� ���� �̹���(�ٿ����¡ ��)
};

// core.cpp
double calculate_gaussian_normal_distribution(double x, double w, double std, double mu);
double make_weight_per_color(int color, int val);
double make_weight_per_saturation(int val, int mu);
double make_weight_per_value(int val, int mu);

void downsize_image(Mat &src, Mat &dst, int downsized_row, int downsized_col);
void update_hue(int pos);
void update_saturation(int pos);
void update_lightness(int pos);
void update_temperature(int pos);
void update_vibrance(int pos);
void update_highlight_saturation(int pos);
void update_highlight_hue(int pos);
void update_shadow_hue(int pos);
void update_shadow_saturation(int pos);
void apply_filter();

void update_tint(int pos);
void update_grain(int pos);
void update_clarity(int pos);
void update_brightness_and_constrast(int brightness_pos, int constrast_pos);
void update_exposure(int pos);
void update_gamma(int pos);
void update_vignette(int pos);

// test.cpp
void mouse_callback(int event, int x, int y, int flags, void *userdata);

void onchange_hue(int pos, void *ptr);
void onchange_saturation(int v, void *ptr);
void onchange_lightness(int v, void *ptr);
void onchange_temperature(int v, void *ptr);
void onchange_vibrance(int v, void *ptr);

void onchange_highlight_saturation(int curPos, void *ptr);
void onchange_highlight_hue(int curPos, void *ptr);
void onchange_shadow_hue(int curPos, void *ptr);
void onchange_shadow_saturation(int curPos, void *ptr);

void on_change_tint(int pos, void *ptr);
void on_change_grain(int pos, void *ptr);
void on_change_clarity(int pos, void *ptr);
void on_change_bright(int pos, void *ptr);
void on_change_constrast(int pos, void *ptr);
void on_change_exposure(int pos, void *ptr);
void on_change_gamma(int pos, void *ptr);
void on_change_vignette(int pos, void *ptr);

extern WorkingImgInfo imginfo;
