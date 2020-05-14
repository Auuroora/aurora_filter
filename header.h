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

// 색 공간 인덱스
typedef enum
{
	B = 0,
	G,
	R,
	H = 0,
	S,
	V
} ColorSpaceIndex;

// 작업중인 모든 변수 다 여기에
class WorkingImgInfo
{
public:
	WorkingImgInfo()
	{
		// split할 벡터 메모리 할당
		this->filter.bgr_filters.resize(3);
		this->filter.bgr_filters.resize(3);
	};

	double min_h, max_h;
	double min_s, max_s;
	double min_v, max_v;
	double min_r, max_r;
	double min_g, max_g;
	double min_b, max_b;

	int row; // 다운사이징 후 사진 가로
	int col; // 다운사이징 후 사진 세로

	Mat downsized_img;	   // 다운사이징 후 이미지
	Mat bgr_img, hsv_img; // bgr이미지, hsv이미지
	Mat res_img;		   // 최종 결과물

	vector<Mat> bgr_split; //bgrImg를 split한 벡터
	vector<Mat> hsv_split; //hsvImg를 split한 벡터
	vector<Mat> res_split;

	// filter
	struct Filter
	{
		Mat diff;		 // 필터 연산을 위한 1채널 행렬
		vector<Mat> diffs;	//필터 연산을 위한 3채널 행렬
		Mat bgr_filter; // bgr변경치가 기록되어 있는 필터
		Mat hsv_filter; // hsv변경치가 기록되어 있는 필터

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
		//struct HSV {
		//	struct Hue {
		//		int	red = 0, orange = 0, yellow = 0,
		//			green = 0, blue = 0, violet = 0;
		//	} hue;

		//	struct Sat {
		//		int	red = 0, orange = 0, yellow = 0,
		//			green = 0, blue = 0, violet = 0;
		//	} sat;

		//	struct Val {
		//		int	red = 0, orange = 0, yellow = 0,
		//			green = 0, blue = 0, violet = 0;
		//	} val;
		//} hsv;

		int temperature = 0;
		int hue;
		int saturation = 0;
		int value = 0;
		int vibrance = 0;
		int highlight = 0;

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
	Mat origin_img; // 변경 불가한 원본 이미지(다운사이징 전)
};

class ParallelModulo : public ParallelLoopBody
{
private:
	Mat &src;
	Mat &dst;
	short *data_src;
	short *data_dst;
	int mod;

public:
	ParallelModulo(Mat &src, Mat &dst, int mod) : src(src), dst(dst), mod(mod)
	{
		data_src = (short *)src.data;
		data_dst = (short *)dst.data;
	}

	virtual void operator()(const Range &range) const CV_OVERRIDE
	{
		for (int r = range.start; r < range.end; r++)
		{
			data_dst[r] = (data_src[r] < 0 ? data_src[r] + mod : data_src[r] % mod);
		}
	}

	ParallelModulo &operator=(const ParallelModulo &)
	{
		return *this;
	};
};

class ParallelMakeWeight : public ParallelLoopBody
{
private:
	Mat &origin;
	Mat &weigh_Matrix;
	double min, max;
	double (*weight_func)(int, int);

public:
	ParallelMakeWeight(Mat &i, Mat &w, double (*wF)(int, int)) : origin(i), weigh_Matrix(w), weight_func(wF)
	{
		cv::minMaxIdx(origin, &min, &max);
	}

	virtual void operator()(const Range &range) const CV_OVERRIDE
	{
		for (int r = range.start; r < range.end; r++)
		{
			weigh_Matrix.data[r] = 10.0; //weight_func((int)origin.data[r], max);
		}
	}

	ParallelMakeWeight &operator=(const ParallelMakeWeight &)
	{
		return *this;
	};
};

// core.cpp
double calculate_gaussian_normal_distribution(double x, double w, double std, double mu);
double make_weight_per_color(int color, int val);
double make_weight_per_saturation(int val, int mu);
double make_weight_per_value(int val, int mu);
void apply_filter();

void update_hue(int pos);
void update_saturation(int pos);
void update_value(int pos);
void update_temperature(int pos);
void update_vibrance(int pos);
void update_highlight_hue(int pos);
void update_highlight_saturation(int pos);
void update_tint(int pos);
void update_grain(int pos);
void update_clarity(int pos);
void update_brightness_and_constrast(int brightness_pos, int constrast_pos);
void update_exposure(int pos);
void update_gamma(int pos);
void update_vignette(int pos);

// test.cpp
void mouse_callback(int event, int x, int y, int flags, void *userdata);
void on_change_hue(int pos, void *ptr);
void on_change_saturation(int v, void *ptr);
void on_change_value(int v, void *ptr);
void on_change_temperature(int v, void *ptr);
void on_change_vibrance(int v, void *ptr);
void on_change_highlight_hue(int curPos, void *ptr);
void on_change_tint(int pos, void *ptr);
void on_change_grain(int pos, void *ptr);
void on_change_clarity(int pos, void *ptr);
void on_change_bright(int pos, void *ptr);
void on_change_constrast(int pos, void *ptr);
void on_change_exposure(int pos, void *ptr);
void on_change_gamma(int pos, void *ptr);
void on_change_vignette(int pos, void *ptr);

// 테스트용
void on_change_color_filter(int curPos, void *ptr);

extern WorkingImgInfo imginfo;