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
typedef enum {
	B = 0, G, R,
	H = 0, S, V
} ColorSpaceIndex;

// �۾����� ��� ���� �� ���⿡
class WorkingImgInfo {
public:
	WorkingImgInfo() {
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

	UMat downsizedImg;		// �ٿ����¡ �� �̹���
	UMat bgrImg, hsvImg;	// bgr�̹���, hsv�̹���
	UMat resImg;			// ���� �����

	// filter
	struct Filter {
		UMat diff;					// ���� ������ ���� ���
		UMat bgr_filter;			// bgr����ġ�� ��ϵǾ� �ִ� ����
		UMat hsv_filter;			// hsv����ġ�� ��ϵǾ� �ִ� ����
		
		UMat clarity_filter;
		UMat clarity_mask;

		UMat gaussian_kernel;

		UMat grain_mask;
		UMat salt_mask;
		UMat pepper_mask;

		UMat exposure_mask;

		vector<UMat> bgr_filters;	// split�� ����(bgr)
		vector<UMat> hsv_filters;	// split�� ����(hsv)
	} filter;

	// �� ����� ����ġ ���
	struct Weight {
		UMat blue, green, red;
		UMat hue, sat, val;
	} weight;

	// trackbar pos
	// ���� Ʈ���� ���� ������ ������
	struct Trackbar {
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
		
		int tint = 0;
		int clarity = 0;
		int exposure = 0;
		int gamma = 0;
		int grain = 0;
		int vignette = 0;

	} trackbar;

	// getter & setter
	Mat getOriginImg() {
		return this->originImg;
	}

	void setOriginImg(Mat img) {
		this->originImg = img.clone();
	}

	UMat getResImg() {
		return this->resImg;
	}

private:
	Mat originImg; // ���� �Ұ��� ���� �̹���(�ٿ����¡ ��)
};

class ParallelModulo : public ParallelLoopBody {
private:
	Mat &src;
	Mat &dst;
	short* dataSrc;
	short* dataDst;
	int mod;

public:
	ParallelModulo(Mat &src, Mat &dst, int mod) : src(src), dst(dst), mod(mod) {
		dataSrc = (short*)src.data;
		dataDst = (short*)dst.data;
	}

	virtual void operator ()(const Range& range) const CV_OVERRIDE {
		for (int r = range.start; r < range.end; r++) {
			dataDst[r] = (dataSrc[r] < 0 ? dataSrc[r] + mod : dataSrc[r] % mod);
		}
	}

	ParallelModulo& operator=(const ParallelModulo &) {
		return *this;
	};
};

class ParallelMakeWeight : public ParallelLoopBody {
private:
	Mat &origin;
	Mat &weighMatrix;
	double min, max;
	double(*weightFunc)(int, int);

public:
	ParallelMakeWeight(Mat &i, Mat &w, double(*wF)(int, int)) : origin(i), weighMatrix(w), weightFunc(wF) {
		cv::minMaxIdx(origin, &min, &max);
	}

	virtual void operator ()(const Range& range) const CV_OVERRIDE {
		for (int r = range.start; r < range.end; r++) {
			weighMatrix.data[r] = 10.0;//weightFunc((int)origin.data[r], max);
		}
	}

	ParallelMakeWeight& operator=(const ParallelMakeWeight &) {
		return *this;
	};
};

// core.cpp
double GND(double x, double w, double std, double mu);
double weightPerColor(int color, int val);
double weightPerSaturation(int val, int mu);
double weightPerValue(int val, int mu);
void updateHue(int pos);
void updateSaturation(int pos);
void updateValue(int pos);
void updateTemperature(int pos);
void updateVibrance();
void updateHighlightSaturation();
void updateHighlightHue();
void applyFilter();

// callback
void mouseCallback(int event, int x, int y, int flags, void *userdata);
void onChangeHue(int pos, void* ptr);
void onChangeSaturation(int v, void* ptr);
void onChangeValue(int v, void* ptr);
void onChangeTemperature(int v, void* ptr);
void onChangeVibrance(int v, void* ptr);
void onChangeHighlight(int curPos, void* ptr);


// �׽�Ʈ��
void onChangeColorFilter(int curPos, void* ptr);

extern WorkingImgInfo imginfo;

/*********************************************************************
*	���� ������ �ڵ�
*********************************************************************/
void update_brightness(int pos);
void update_constrast(int brightnessValue, int constrastValue);
void update_clarity(int pos);
void upadate_exposure(int pos);
void update_gamma(int pos);
void update_grain(int pos);
void update_vignette(int pos);
void update_tint(int pos);