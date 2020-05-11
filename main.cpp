#include "define.h"
#include "header.h"

using namespace cv;
using namespace std;

WorkingImgInfo imginfo;

void init(Mat &img)
{
	// save original Image
	imginfo.set_origin_img(img);

	// downsizing
	imginfo.downsized_img = img.clone().getUMat(ACCESS_RW);
	imginfo.row = imginfo.downsized_img.rows;
	imginfo.col = imginfo.downsized_img.cols;
	// TO DO

	// convert to 3 channels(BGRA -> BGR)
	if (imginfo.downsized_img.channels() == 4)
	{
		cv::cvtColor(imginfo.downsized_img, imginfo.downsized_img, COLOR_BGRA2BGR);
	}

	// setting img
	//imginfo.bgr_img = imginfo.downsized_img.clone();
	imginfo.bgr_img = imginfo.downsized_img.clone();
	cv::cvtColor(imginfo.bgr_img, imginfo.hsv_img, COLOR_BGR2HSV);
	cv::split(imginfo.bgr_img, imginfo.filter.bgr_filters);
	cv::split(imginfo.hsv_img, imginfo.filter.hsv_filters);

	//split img
	cv::split(imginfo.bgr_img, imginfo.bgr_split);
	cv::split(imginfo.hsv_img, imginfo.hsv_split);

	//*******************************************************************************************************

	// Gamma
	imginfo.filter.hsv_filters[ColorSpaceIndex::V].convertTo(imginfo.filter.gamma_mask, CV_32F);
	cv::multiply(1. / 255, imginfo.filter.gamma_mask, imginfo.filter.gamma_mask);

	//Clarity
	cv::bilateralFilter(imginfo.bgr_img, imginfo.filter.clarity_filter, DISTANCE, SIGMA_COLOR, SIGMA_SPACE);
	imginfo.filter.clarity_mask = UMat::zeros(imginfo.col, imginfo.row, CV_16SC3);

	//Vignette
	Mat kernel_x, kernel_y, kernel_res;
	kernel_x = cv::getGaussianKernel(imginfo.row, 1000);
	kernel_y = cv::getGaussianKernel(imginfo.col, 1000);
	cv::transpose(kernel_x, kernel_x);
	kernel_res = (kernel_y * kernel_x);
	cv::normalize(kernel_res, kernel_res, 0, 1, NORM_MINMAX);
	imginfo.filter.gaussian_kernel = kernel_res.getUMat(ACCESS_RW);

	//Grain
	imginfo.filter.grain_mask = UMat::zeros(imginfo.col, imginfo.row, CV_16S);

	cv::randu(imginfo.filter.grain_mask, Scalar(-20), Scalar(20));
	// imginfo.filter.salt_mask = UMat(imginfo.col, imginfo.row, CV_8U);
	// imginfo.filter.pepper_mask = UMat(imginfo.col, imginfo.row, CV_8U);

	//Exposure
	imginfo.filter.exposure_mask = UMat::ones(imginfo.col, imginfo.row, CV_8UC1);

	//*******************************************************************************************************

	// cal minmax
	cv::minMaxIdx(imginfo.filter.bgr_filters[ColorSpaceIndex::B], &imginfo.min_b, &imginfo.max_b);
	cv::minMaxIdx(imginfo.filter.bgr_filters[ColorSpaceIndex::G], &imginfo.min_g, &imginfo.max_g);
	cv::minMaxIdx(imginfo.filter.bgr_filters[ColorSpaceIndex::R], &imginfo.min_r, &imginfo.max_r);

	cv::minMaxIdx(imginfo.filter.hsv_filters[ColorSpaceIndex::H], &imginfo.min_h, &imginfo.max_h);
	cv::minMaxIdx(imginfo.filter.hsv_filters[ColorSpaceIndex::S], &imginfo.min_s, &imginfo.max_s);
	cv::minMaxIdx(imginfo.filter.hsv_filters[ColorSpaceIndex::V], &imginfo.min_v, &imginfo.max_v);

	// init weight and diff matrix
	imginfo.weight.hue = UMat::ones(imginfo.row, imginfo.col, CV_32F);
	imginfo.weight.sat = UMat::ones(imginfo.row, imginfo.col, CV_32F);
	imginfo.weight.val = UMat::ones(imginfo.row, imginfo.col, CV_32F);

	imginfo.filter.bgr_filters[ColorSpaceIndex::B] = UMat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.bgr_filters[ColorSpaceIndex::G] = UMat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.bgr_filters[ColorSpaceIndex::R] = UMat::zeros(imginfo.row, imginfo.col, CV_16S);

	imginfo.filter.hsv_filters[ColorSpaceIndex::H] = UMat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.hsv_filters[ColorSpaceIndex::S] = UMat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.hsv_filters[ColorSpaceIndex::V] = UMat::zeros(imginfo.row, imginfo.col, CV_16S);

	imginfo.filter.diff = UMat::zeros(imginfo.row, imginfo.col, CV_16S);

	// make weight matrix
	// TO DO

	// TO DO
}

int main()
{
	/*********************************************************************
	*	OpenCL Test
	*********************************************************************/
	// OpenCL을 사용할 수 있는지 테스트
	if (!ocl::haveOpenCL())
	{
		cout << "에러 : OpenCL을 사용할 수 없는 시스템입니다." << endl;
		return -1;
	}

	// 컨텍스트 생성
	ocl::Context context;
	if (!context.create(ocl::Device::TYPE_GPU))
	{
		cout << " 에러 : 컨텍스트를 생성할 수 없습니다." << endl;
		return -1;
	}

	// GPU 장치 정보
	cout << context.ndevices() << " GPU device (s) detected " << endl;
	for (size_t i = 0; i < context.ndevices(); i++)
	{
		ocl::Device device = context.device(i);
		cout << " - Device " << i << " --- " << endl;
		cout << " Name : " << device.name() << endl;
		cout << " Availability : " << device.available() << endl;
		cout << "Image Support : " << device.imageSupport() << endl;
		cout << " OpenCL C version : " << device.OpenCL_C_Version() << endl;
	}

	ocl::Device(context.device(0));
	ocl::setUseOpenCL(true);

	/*********************************************************************
	*	Init
	*********************************************************************/
	Mat inputImg = imread("3024x3024.jpg", IMREAD_COLOR);
	if (inputImg.empty())
	{
		cout << "Image Open Failed" << endl;
		return -1;
	}

	init(inputImg);

	/*********************************************************************
	*	Make Window
	*********************************************************************/
	cv::namedWindow(TEST_WINDOW, WINDOW_NORMAL);
	cv::resizeWindow(TEST_WINDOW, 400, 600);
	//setMouseCallback(TEST_WINDOW, mouseCallback, &imginfo.resImg);

	cv::namedWindow(SET_WINDOW, WINDOW_NORMAL);
	cv::resizeWindow(SET_WINDOW, 1000, 200);
	cv::imshow(TEST_WINDOW, imginfo.bgr_img);

	/*********************************************************************
	*	Make Trackbar
	*********************************************************************/
	//// HUE
	//createTrackbar("Hue", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, onChangeHue);
	//setTrackbarPos("Hue", SET_WINDOW, TRACKBAR_MID);

	//// Saturation
	//createTrackbar("Sat", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, onChangeSaturation);
	//setTrackbarPos("Sat", SET_WINDOW, TRACKBAR_MID);

	//// Value
	cv::createTrackbar("Value", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_value);
	cv::setTrackbarPos("Value", SET_WINDOW, TRACKBAR_MID);

	// Temperature
	cv::createTrackbar("Vibrance", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_vibrance);
	cv::setTrackbarPos("Vibrance", SET_WINDOW, TRACKBAR_MID);

	//// Vibrance
	//createTrackbar("Vibrance", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, onChangeVibrance);
	//setTrackbarPos("Vibrance", SET_WINDOW, TRACKBAR_MID);

	//// ColorFilter
	//createTrackbar("R", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, onChangeColorFilter);
	//setTrackbarPos("R", SET_WINDOW, TRACKBAR_MIN);
	//createTrackbar("G", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, onChangeColorFilter);
	//setTrackbarPos("G", SET_WINDOW, TRACKBAR_MIN);
	//createTrackbar("B", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, onChangeColorFilter);
	//setTrackbarPos("B", SET_WINDOW, TRACKBAR_MIN);

	//// Highlight
	//createTrackbar("H", SET_WINDOW, TRACKBAR_MIN, HUE_MAX, onChangeHighlight);
	//setTrackbarPos("H", SET_WINDOW, TRACKBAR_MIN);
	//createTrackbar("S", SET_WINDOW, TRACKBAR_MIN, SAT_MAX, onChangeHighlight);
	//setTrackbarPos("S", SET_WINDOW, TRACKBAR_MIN);

	while (waitKey(0) != 27)
		;
	cv::destroyAllWindows();
	return 0;
}
