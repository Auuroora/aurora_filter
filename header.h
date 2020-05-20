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

// 색 공간 인덱스
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

// 작업중인 모든 변수 다 여기에
class WorkingImgInfo
{
public:
	WorkingImgInfo(){

	};

	int row; // 다운사이징 후 사진 가로
	int col; // 다운사이징 후 사진 세로
	int changed_color_space = 0;

	/***********************************************************************************************/
	struct Image
	{
		Mat downsized;			 // 다운사이징 후 이미지
		Mat bgr, hls, hsv, res;	 // bgr이미지, hsv이미지, 최종 결과물
		vector<Mat> bgr_origins; // split한 벡터(bgr)
		vector<Mat> hls_origins; // split한 벡터(hls)
		vector<Mat> hsv_origins; // split한 벡터(hsv)
		vector<Mat> res_split;
	} image;

	/***********************************************************************************************/

	// filter
	struct Filter
	{
		Mat diff;		   // 필터 연산을 위한 1채널 행렬
		vector<Mat> diffs; //필터 연산을 위한 3채널 행렬
		Mat bgr_filter;	   // bgr변경치가 기록되어 있는 필터
		Mat hsv_filter;	   // hsv변경치가 기록되어 있는 필터
		Mat hls_filter;	   // hsv변경치가 기록되어 있는 필터

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

		vector<Mat> bgr_filters; // split한 벡터(bgr)
		vector<Mat> hls_filters; // split한 벡터(hls)
		vector<Mat> hsv_filters; // split한 벡터(hsv)
	} filter;

	// 색 검출용 가중치 행렬
	struct Weight
	{
		Mat blue, green, red;
		Mat hue, sat, val;
	} weight;

	// trackbar pos
	// 현재 트랙바 상태 저장한 변수들
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
		return this->image.res;
	}

private:
	Mat origin_img; // 변경 불가한 원본 이미지(다운사이징 전)
};

void downsize_image(Mat &src, Mat &dst, int downsizing_row, int downsizing_col);
void apply_filter();
void update_hue(int pos);
void update_saturation(int pos);
void update_lightness(int pos);
void update_temperature(int pos);
void update_vibrance(int pos);
void update_highlight_saturation(int pos);
void update_highlight_hue(int pos);
void update_shadow_hue(int pos);
void update_shadow_saturation(int pos);

void update_tint(int pos);
void update_clarity(int pos);
void update_brightness_and_constrast(int brightness_pos, int constrast_pos);
void update_exposure(int pos);
void update_gamma(int pos);
void update_grain(int pos);
void update_vignette(int pos);

// test.cpp
void mouse_callback(int event, int x, int y, int flags, void *userdata);

void on_change_hue(int pos, void *ptr);
void on_change_saturation(int v, void *ptr);
void on_change_lightness(int v, void *ptr);
void on_change_temperature(int v, void *ptr);
void on_change_vibrance(int v, void *ptr);

void on_change_highlight_saturation(int curPos, void *ptr);
void on_change_highlight_hue(int curPos, void *ptr);
void on_change_shadow_hue(int curPos, void *ptr);
void on_change_shadow_saturation(int curPos, void *ptr);

void on_change_tint(int pos, void *ptr);
void on_change_clarity(int pos, void *ptr);
void on_change_bright(int pos, void *ptr);
void on_change_constrast(int pos, void *ptr);
void on_change_exposure(int pos, void *ptr);
void on_change_gamma(int pos, void *ptr);
void on_change_grain(int pos, void *ptr);
void on_change_vignette(int pos, void *ptr);

extern WorkingImgInfo imginfo;
