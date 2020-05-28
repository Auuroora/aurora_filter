#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "define.h"

// 색 공간 인덱스
enum class BGR {
	B, G, R
};

enum class HSV {
	H, S, V
};

enum class HLS {
	H, L, S
};

// Utils
double GND(double x, double w, double std, double mu);
void downsize_image(cv::Mat &src, cv::Mat &dst, int width, int height);
void mouse_callback(int event, int x, int y, int flags, void *userdata);
double weight_per_saturation(int val, int mu);
double weight_per_value(int val, int mu);

cv::Mat cut_image(cv::Mat src, int start_x, int start_y, int end_x, int end_y);
cv::Mat get_watermarked_image(cv::Mat src_img, cv::Mat src_logo, int width = 0, int height = 0);
cv::Mat get_preview_image(
	cv::Mat& src_img, cv::Mat src_logo,
	int hue = 0, int saturation = 0, int lightness = 0, int vibrance = 0,
	int highlight_hue = 0, int highlight_sat = 0, int shadow_hue = 0, int shadow_sat = 0,
	int temperature = 0, int tint = 0, int brightness = 0, int grain = 0,
	int clarity = 0, int exposure = 0, int gamma = 0, int vignette = 0, int constrast = 0,
	int width = 0, int height = 0 /* for downsizing */
);

void on_change_hue(int pos, void* ptr);
void on_change_saturation(int v, void* ptr);
void on_change_lightness(int v, void* ptr);
void on_change_vibrance(int v, void* ptr);
void on_change_highlight_saturation(int curPos, void* ptr);
void on_change_highlight_hue(int curPos, void* ptr);
void on_change_shadow_hue(int curPos, void* ptr);
void on_change_shadow_saturation(int curPos, void* ptr);
void on_change_temperature(int v, void* ptr);
void on_change_bright(int pos, void *ptr);
void on_change_tint(int pos, void *ptr);
void on_change_constrast(int pos, void *ptr);
void on_change_clarity(int pos, void *ptr);
void on_change_bright(int pos, void *ptr);
void on_change_exposure(int pos, void *ptr);
void on_change_gamma(int pos, void *ptr);
void on_change_grain(int pos, void *ptr);
void on_change_vignette(int pos, void *ptr);

// 작업중인 모든 변수 다 여기에
class WorkingImgInfo {
public:
	WorkingImgInfo() {
		
	}

	/*********************************************************************
	*	variable and struct
	*********************************************************************/
	int width;									// 다운사이징 후 사진 세로(height가 세로 맞음)
	int height;									// 다운사이징 후 사진 가로

	struct Image {
		cv::Mat downsized;						// 다운사이징 후 이미지
		cv::Mat bgr, hls, hsv, res;				// bgr이미지, hsv이미지, 최종 결과물
		std::vector<cv::Mat> bgr_origins;		// split한 벡터(bgr)
		std::vector<cv::Mat> hls_origins;		// split한 벡터(hls)
		std::vector<cv::Mat> hsv_origins;		// split한 벡터(hsv)
	} image;

	struct Filter {
		cv::Mat diff;							// 필터 연산을 위한 행렬
		cv::Mat bgr_filter;						// bgr변경치가 기록되어 있는 필터
		cv::Mat hls_filter;						// hls변경치가 기록되어 있는 필터
		cv::Mat hsv_filter;						// hsv변경치가 기록되어 있는 필터

		cv::Mat clarity_filter;
		cv::Mat clarity_mask_U;
		cv::Mat clarity_mask_S;
		std::vector<cv::Mat> clarity_mask_split;

		cv::Mat gaussian_kernel;
		cv::Mat gamma_mask;
		cv::Mat grain_mask;
		cv::Mat salt_mask;
		cv::Mat pepper_mask;
		cv::Mat exposure_mask;

		std::vector<cv::Mat> bgr_filters;		// split한 벡터(bgr)
		std::vector<cv::Mat> hls_filters;		// split한 벡터(hls)
		std::vector<cv::Mat> hsv_filters;		// split한 벡터(hsv)
	} filter;

	// 색 검출용 가중치 행렬
	struct Weight {
		cv::Mat blue, green, red;
		cv::Mat hue, saturation, lightness;
	} weight;

	// trackbar pos
	// 현재 트랙바 상태 저장한 변수들
	struct Trackbar {
		int temperature;
		int hue;
		int saturation;
		int lightness;
		int vibrance;
		int highlight_hue;
		int highlight_sat;
		int shadow_hue;
		int shadow_sat;
		int tint;
		int clarity;
		int exposure;
		int gamma;
		int grain;
		int vignette;
		int constrast;
		int brightness;
	} trackbar;

	/*********************************************************************
	*	method
	*********************************************************************/
	void update_hue(int pos);
	void update_saturation(int pos);
	void update_lightness(int pos);
	void update_vibrance(int pos);
	void update_highlight_saturation(int pos);
	void update_highlight_hue(int pos);
	void update_shadow_hue(int pos);
	void update_shadow_saturation(int pos);
	void update_temperature(int pos);
	void update_brightness_and_constrast(int brightness_pos, int constrast_pos);
	void update_tint(int pos);
	void update_clarity(int pos);
	void update_exposure(int pos);
	void update_gamma(int pos);
	void update_grain(int pos);
	void update_vignette(int pos);
	void apply_filter();

	/* first initialize */
	void init_all(cv::Mat& img, int width, int height) {
		this->originImg = img.clone();
		this->init_image(width, height);
		this->init_filter();
		this->init_weight();
		this->init_trackbar(0);
	}

	/* image initialize */
	void init_image(int width, int height) {
		/* downsizing */
		if (!width || !height) {
			width = this->originImg.size().width;
			height = this->originImg.size().height;
		}

		downsize_image(this->originImg, this->image.downsized, width, height);
		this->width = this->image.downsized.size().width;
		this->height = this->image.downsized.size().height;

		/* convert */
		this->image.bgr = this->image.downsized.clone();
		if (this->image.bgr.channels() == 4) {
			cv::cvtColor(this->image.bgr, this->image.bgr, cv::COLOR_BGRA2BGR);
		}
		cv::cvtColor(this->image.bgr, this->image.hls, cv::COLOR_BGR2HLS);
		cv::cvtColor(this->image.bgr, this->image.hsv, cv::COLOR_BGR2HSV);
	}

	/* filter matrix initialize */
	void init_filter() {
		/* split(original) */
		cv::split(this->image.bgr, this->image.bgr_origins);
		cv::split(this->image.hls, this->image.hls_origins);
		cv::split(this->image.hsv, this->image.hsv_origins);

		/* delete 0 in hls,hsv image */
		cv::Mat mask;
		cv::inRange(this->image.hls_origins[HLSINDEX::L], 0, 0, mask);
		this->image.hls_origins[HLSINDEX::L].setTo(1, mask);
		cv::inRange(this->image.hls_origins[HLSINDEX::S], 0, 0, mask);
		this->image.hls_origins[HLSINDEX::S].setTo(1, mask);
		cv::inRange(this->image.hsv_origins[HSVINDEX::S], 0, 0, mask);
		this->image.hsv_origins[HSVINDEX::S].setTo(1, mask);
		cv::inRange(this->image.hsv_origins[HSVINDEX::V], 0, 0, mask);
		this->image.hsv_origins[HSVINDEX::V].setTo(1, mask);

		/* initialize filter matrix */
		this->filter.bgr_filter = cv::Mat::zeros(this->height, this->width, CV_16SC3);
		this->filter.hls_filter = cv::Mat::zeros(this->height, this->width, CV_16SC3);
		this->filter.hsv_filter = cv::Mat::zeros(this->height, this->width, CV_16SC3);
		this->filter.diff = cv::Mat::zeros(this->height, this->width, CV_16S);

		cv::split(this->filter.bgr_filter, this->filter.bgr_filters);
		cv::split(this->filter.hls_filter, this->filter.hls_filters);
		cv::split(this->filter.hsv_filter, this->filter.hsv_filters);

		/*****************************************************************************/
		// Gamma
		this->filter.hsv_filters[HSVINDEX::V].convertTo(this->filter.gamma_mask, CV_32F);
		cv::multiply(1.0 / 255.0, this->filter.gamma_mask, this->filter.gamma_mask);

		//Clarity
		cv::bilateralFilter(this->image.bgr, this->filter.clarity_filter, DISTANCE, SIGMA_COLOR, SIGMA_SPACE);
		this->filter.clarity_mask_U = cv::Mat::zeros(this->height, this->width, CV_8UC3);
		this->filter.clarity_mask_S = cv::Mat::zeros(this->height, this->width, CV_16SC3);

		//Vignette
		cv::Mat kernel_x, kernel_x_transpose, kernel_y, kernel_res;
		kernel_x = cv::getGaussianKernel(this->width, 1000, CV_32F);
		kernel_y = cv::getGaussianKernel(this->height, 1000, CV_32F);
		cv::transpose(kernel_x, kernel_x_transpose);
		kernel_res = (kernel_y * kernel_x_transpose);
		cv::normalize(kernel_res, kernel_res, 0, 1, cv::NORM_MINMAX);
		cv::subtract(1, kernel_res, kernel_res);
		kernel_res = cv::abs(kernel_res);
		cv::multiply(125, kernel_res, kernel_res);
		kernel_res.convertTo(kernel_res, CV_16S);
		this->filter.gaussian_kernel = kernel_res.clone();

		kernel_x.deallocate();
		kernel_x_transpose.deallocate();
		kernel_y.deallocate();
		kernel_res.deallocate();

		//Grain
		this->filter.grain_mask = cv::Mat::zeros(this->height, this->width, CV_32F);
		cv::randu(this->filter.grain_mask, cv::Scalar(-20), cv::Scalar(20));

		//Exposure
		this->filter.exposure_mask = cv::Mat::ones(this->height, this->width, CV_8UC1);
	}

	/* make weight matrix */
	void init_weight() {
		// TO DO
	}

	/* set trackbar pos */
	void init_trackbar(int pos) {
		this->trackbar.temperature = pos;
		this->trackbar.hue = pos;
		this->trackbar.saturation = pos;
		this->trackbar.lightness = pos;
		this->trackbar.vibrance = pos;
		this->trackbar.highlight_hue = pos;
		this->trackbar.highlight_sat = pos;
		this->trackbar.shadow_hue = pos;
		this->trackbar.shadow_sat = pos;
		this->trackbar.tint = pos;
		this->trackbar.clarity = pos;
		this->trackbar.exposure = pos;
		this->trackbar.gamma = pos;
		this->trackbar.grain = pos;
		this->trackbar.vignette = pos;
		this->trackbar.constrast = pos;
		this->trackbar.brightness = pos;
	}

	cv::Mat get_origin_img() {
		return this->originImg;
	}

	cv::Mat get_res_img() {
		return this->image.res;
	}
private:
	cv::Mat originImg; // 변경 불가한 원본 이미지(다운사이징 전)
};

class ParallelModulo : public cv::ParallelLoopBody {
private:
	cv::Mat &src;
	cv::Mat &dst;
	short* dataSrc;
	short* dataDst;
	int mod;

public:
	ParallelModulo(cv::Mat &src, cv::Mat &dst, int mod) : src(src), dst(dst), mod(mod) {
		dataSrc = (short*)src.data;
		dataDst = (short*)dst.data;
	}

	virtual void operator ()(const cv::Range& range) const CV_OVERRIDE {
		for (int r = range.start; r < range.end; r++) {
			dataDst[r] = (dataSrc[r] < 0 ? dataSrc[r] + mod : dataSrc[r] % mod);
		}
	}

	ParallelModulo& operator=(const ParallelModulo &) {
		return *this;
	};
};

class ParallelMakeWeight : public cv::ParallelLoopBody {
private:
	cv::Mat& src;
	cv::Mat& weight_mat;
	double min, max;
	double(*weight_func)(int, int);

public:
	ParallelMakeWeight(cv::Mat &i, cv::Mat &w, double(*wF)(int, int))
		: src(i), weight_mat(w), weight_func(wF) {
		cv::minMaxIdx(src, &min, &max);
	}

	virtual void operator ()(const cv::Range& range) const CV_OVERRIDE {
		for (int r = range.start; r < range.end; r++) {
			weight_mat.data[r] = weight_func((int)src.data[r], max);
		}
	}

	ParallelMakeWeight& operator=(const ParallelMakeWeight &) {
		return *this;
	};
};

/* global variable */
extern WorkingImgInfo imginfo;
extern cv::Mat main_logo;