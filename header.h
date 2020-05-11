#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif // _WIN32

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

// �� ���� �ε���
typedef enum
{
	B = 0,
	G,
	R,
	H = 0,
	S,
	V
} ColorSpaceIndex;

// �۾����� ��� ���� �� ���⿡
class WorkingImgInfo
{
public:
	WorkingImgInfo()
	{
		// split�� ���� �޸� �Ҵ�
		this->filter.bgr_filters.resize(3);
		this->filter.bgr_filters.resize(3);
	};

	double min_h, max_h;
	double min_s, max_s;
	double min_v, max_v;
	double min_r, max_r;
	double min_g, max_g;
	double min_b, max_b;

	int row; // �ٿ����¡ �� ���� ����
	int col; // �ٿ����¡ �� ���� ����

	UMat downsized_img;	   // �ٿ����¡ �� �̹���
	UMat bgr_img, hsv_img; // bgr�̹���, hsv�̹���
	UMat res_img;		   // ���� �����

	vector<UMat> bgr_split; //bgrImg�� split�� ����
	vector<UMat> hsv_split; //hsvImg�� split�� ����

	// filter
	struct Filter
	{
		UMat diff;		 // ���� ������ ���� ���
		UMat bgr_filter; // bgr����ġ�� ��ϵǾ� �ִ� ����
		UMat hsv_filter; // hsv����ġ�� ��ϵǾ� �ִ� ����

		UMat clarity_filter;
		UMat clarity_mask;

		UMat gaussian_kernel;

		UMat gamma_mask;

		UMat grain_mask;
		UMat salt_mask;
		UMat pepper_mask;

		UMat exposure_mask;

		vector<UMat> bgr_filters; // split�� ����(bgr)
		vector<UMat> hsv_filters; // split�� ����(hsv)
	} filter;

	// �� ����� ����ġ ���
	struct Weight
	{
		UMat blue, green, red;
		UMat hue, sat, val;
	} weight;

	// trackbar pos
	// ���� Ʈ���� ���� ������ ������
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

	UMat getResImg()
	{
		return this->res_img;
	}

private:
	Mat origin_img; // ���� �Ұ��� ���� �̹���(�ٿ����¡ ��)
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
	Mat &weigh_matrix;
	double min, max;
	double (*weight_func)(int, int);

public:
	ParallelMakeWeight(Mat &i, Mat &w, double (*wF)(int, int)) : origin(i), weigh_matrix(w), weight_func(wF)
	{
		cv::minMaxIdx(origin, &min, &max);
	}

	virtual void operator()(const Range &range) const CV_OVERRIDE
	{
		for (int r = range.start; r < range.end; r++)
		{
			weigh_matrix.data[r] = 10.0; //weight_func((int)origin.data[r], max);
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
void update_hue(int pos);
void update_saturation(int pos);
void update_value(int pos);
void update_temperature(int pos);
void update_vibrance(int pos);
void update_highlight_hue(int pos);
void update_highlight_saturation(int pos);
void apply_filter();

// callback
void mouse_callback(int event, int x, int y, int flags, void *userdata);
void on_change_hue(int pos, void *ptr);
void on_change_saturation(int v, void *ptr);
void on_change_value(int v, void *ptr);
void on_change_temperature(int v, void *ptr);
void on_change_vibrance(int v, void *ptr);
void on_change_highlight(int curPos, void *ptr);

// �׽�Ʈ��
void on_change_color_filter(int curPos, void *ptr);

extern WorkingImgInfo imginfo;

/*********************************************************************
*	���� ������ �ڵ�
*********************************************************************/

void update_brightness_constrast(int brightnessValue, int constrastValue);
void update_exposure(int pos);
void update_gamma(int pos);
void update_grain(int pos);
void update_vignette(int pos);
void update_tint(int pos);