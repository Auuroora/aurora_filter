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
	imginfo.downsized_img = img.clone();			//.getMat(ACCESS_RW);
	imginfo.row = imginfo.downsized_img.rows;		//height
	imginfo.col = imginfo.downsized_img.cols;		//width
	// TO DO

	// convert to 3 channels(BGRA -> BGR)
	if (imginfo.downsized_img.channels() == 4)
	{
		cv::cvtColor(imginfo.downsized_img, imginfo.downsized_img, COLOR_BGRA2BGR);
	}

	// setting img
	//imginfo.bgr_img = imginfo.downsized_img.clone();
	imginfo.bgr_img = imginfo.downsized_img.clone();
	imginfo.res_img = imginfo.downsized_img.clone();
	cv::cvtColor(imginfo.bgr_img, imginfo.hsv_img, COLOR_BGR2HSV);
	cv::split(imginfo.bgr_img, imginfo.filter.bgr_filters);
	cv::split(imginfo.hsv_img, imginfo.filter.hsv_filters);

	//split img
	cv::split(imginfo.downsized_img, imginfo.bgr_split);
	cv::split(imginfo.hsv_img, imginfo.hsv_split);

	//*******************************************************************************************************

	// Gamma
	imginfo.filter.hsv_filters[ColorSpaceIndex::V].convertTo(imginfo.filter.gamma_mask, CV_32F);
	cv::multiply(1. / 255, imginfo.filter.gamma_mask, imginfo.filter.gamma_mask);

	//Clarity
	cv::bilateralFilter(imginfo.bgr_img, imginfo.filter.clarity_filter, DISTANCE, SIGMA_COLOR, SIGMA_SPACE);
	imginfo.filter.clarity_mask_U = Mat::zeros(imginfo.row, imginfo.col, CV_8UC3);
	imginfo.filter.clarity_mask_S = Mat::zeros(imginfo.row, imginfo.col, CV_16SC3);


	//Vignette
	Mat kernel_x,kernel_x_transpose, kernel_y, kernel_res;
	kernel_x = cv::getGaussianKernel(imginfo.col, 1000);
	kernel_y = cv::getGaussianKernel(imginfo.row, 1000);
	cv::transpose(kernel_x, kernel_x_transpose);
	kernel_res = (kernel_y * kernel_x_transpose);
	cv::normalize(kernel_res, kernel_res, 0, 1, NORM_MINMAX);
	imginfo.filter.gaussian_kernel = kernel_res.clone();				//getUMat(cv::ACCESS_RW);
	imginfo.filter.gaussian_kernel=kernel_res.clone();
	kernel_x.deallocate();
	kernel_x_transpose.deallocate();
	kernel_y.deallocate();
	kernel_res.deallocate();


	//Grain
	imginfo.filter.grain_mask = Mat::zeros(imginfo.row, imginfo.col, CV_32F);

	cv::randu(imginfo.filter.grain_mask, Scalar(-20), Scalar(20));
	// imginfo.filter.salt_mask = Mat(imginfo.col, imginfo.row, CV_8U);
	// imginfo.filter.pepper_mask = Mat(imginfo.col, imginfo.row, CV_8U);

	//Exposure
	imginfo.filter.exposure_mask = Mat::ones(imginfo.row, imginfo.col, CV_8UC1);

	//*******************************************************************************************************

	// cal minmax
	cv::minMaxIdx(imginfo.filter.bgr_filters[ColorSpaceIndex::B], &imginfo.min_b, &imginfo.max_b);
	cv::minMaxIdx(imginfo.filter.bgr_filters[ColorSpaceIndex::G], &imginfo.min_g, &imginfo.max_g);
	cv::minMaxIdx(imginfo.filter.bgr_filters[ColorSpaceIndex::R], &imginfo.min_r, &imginfo.max_r);

	cv::minMaxIdx(imginfo.filter.hsv_filters[ColorSpaceIndex::H], &imginfo.min_h, &imginfo.max_h);
	cv::minMaxIdx(imginfo.filter.hsv_filters[ColorSpaceIndex::S], &imginfo.min_s, &imginfo.max_s);
	cv::minMaxIdx(imginfo.filter.hsv_filters[ColorSpaceIndex::V], &imginfo.min_v, &imginfo.max_v);

	// init weight and diff matrix
	imginfo.weight.hue = Mat::ones(imginfo.row, imginfo.col, CV_32F);
	imginfo.weight.sat = Mat::ones(imginfo.row, imginfo.col, CV_32F);
	imginfo.weight.val = Mat::ones(imginfo.row, imginfo.col, CV_32F);

	imginfo.filter.bgr_filters[ColorSpaceIndex::B] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.bgr_filters[ColorSpaceIndex::G] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.bgr_filters[ColorSpaceIndex::R] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);

	imginfo.filter.hsv_filters[ColorSpaceIndex::H] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.hsv_filters[ColorSpaceIndex::S] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.hsv_filters[ColorSpaceIndex::V] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);

	imginfo.filter.diff = Mat::zeros(imginfo.row, imginfo.col, CV_16S);

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
	Mat inputImg = imread("2400x1600.jpg", IMREAD_COLOR);
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

	// Grain
	cv::createTrackbar("Grain", SET_WINDOW, GRAIN_MIN, GRAIN_MAX, on_change_grain);
	cv::setTrackbarPos("Grain", SET_WINDOW, GRAIN_MID);

	//Clarity
	cv::createTrackbar("Vignette", SET_WINDOW, VIGNETTE_MIN, VIGNETTE_MAX, on_change_vignette);
	cv::setTrackbarPos("Vignette", SET_WINDOW, VIGNETTE_MID);

	//Clarity
	cv::createTrackbar("Clarity", SET_WINDOW, CLARITY_MIN, CLARITY_MAX, on_change_clarity);
	cv::setTrackbarPos("Clarity", SET_WINDOW, CLARITY_MID);

	//Exposure
	cv::createTrackbar("Exposure", SET_WINDOW, EXPOSURE_MIN, EXPOSURE_MAX, on_change_exposure);
	cv::setTrackbarPos("Exposure", SET_WINDOW, EXPOSURE_MID);

	//Gamma
	cv::createTrackbar("Gamma", SET_WINDOW, GAMMA_MIN, GAMMA_MAX, on_change_gamma);
	cv::setTrackbarPos("Gamma", SET_WINDOW, GAMMA_MID);

	//B&C
	cv::createTrackbar("BRIGHTNESS", SET_WINDOW, BRIGHTNESS_MIN, BRIGHTNESS_MAX, on_change_bright);
	cv::setTrackbarPos("BRIGHTNESS", SET_WINDOW, BRIGHTNESS_MID);
	cv::createTrackbar("CONSTRAST", SET_WINDOW, CONSTRAST_MIN, CONSTRAST_MAX, on_change_constrast);
	cv::setTrackbarPos("CONSTRAST", SET_WINDOW, CONSTRAST_MID);

	// // ColorFilter
	// createTrackbar("R", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, onChangeColorFilter);
	// setTrackbarPos("R", SET_WINDOW, TRACKBAR_MIN);
	//createTrackbar("G", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, onChangeColorFilter);
	//setTrackbarPos("G", SET_WINDOW, TRACKBAR_MIN);
	//createTrackbar("B", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, onChangeColorFilter);
	//setTrackbarPos("B", SET_WINDOW, TRACKBAR_MIN);

	//// Highlight
	//createTrackbar("H", SET_WINDOW, TRACKBAR_MIN, HUE_MAX, onChangeHighlight);
	//setTrackbarPos("H", SET_WINDOW, TRACKBAR_MIN);
	//createTrackbar("S", SET_WINDOW, TRACKBAR_MIN, SAT_MAX, onChangeHighlight);
	//setTrackbarPos("S", SET_WINDOW, TRACKBAR_MIN);

	while (waitKey(0) != 27);
	cv::destroyAllWindows();
	return 0;
}
