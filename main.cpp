#include "define.h"
#include "header.h"

using namespace cv;
using namespace std;
using namespace BGR;
using namespace HSV;
using namespace HLS;

WorkingImgInfo imginfo;

void init(Mat &img)
{
	/*********************************************************************
	*	convert and setting
	*********************************************************************/
	/* save original Image */
	imginfo.set_origin_img(img);

	/* downsizing */

	//downsizing(img, imginfo.image.downsized, imginfo.row, imginfo.col);

	imginfo.image.downsized = img.clone();
	imginfo.row = imginfo.image.downsized.rows;
	imginfo.col = imginfo.image.downsized.cols;

	/* convert to 3 channels(BGRA -> BGR) */
	if (imginfo.image.downsized.channels() == 4) {
		cv::cvtColor(imginfo.image.downsized, imginfo.image.downsized, COLOR_BGRA2BGR);
	}

	/*********************************************************************
	*	variable initialize
	*********************************************************************/
	/* setting img */
	imginfo.image.bgr = imginfo.image.downsized.clone();
	cv::cvtColor(imginfo.image.bgr, imginfo.image.hls, COLOR_BGR2HLS);
	cv::cvtColor(imginfo.image.bgr, imginfo.image.hsv, COLOR_BGR2HSV);

	cv::split(imginfo.image.bgr, imginfo.image.bgr_origins);
	cv::split(imginfo.image.hls, imginfo.image.hls_origins);
	cv::split(imginfo.image.hsv, imginfo.image.hsv_origins);

	Mat mask;
	cv::inRange(imginfo.image.hls_origins[HLSIndex::S], 0, 0, mask);
	imginfo.image.hls_origins[HLSIndex::S].setTo(1, mask);
	cv::inRange(imginfo.image.hls_origins[HLSIndex::L], 0, 0, mask);
	imginfo.image.hls_origins[HLSIndex::L].setTo(1, mask);

	cv::inRange(imginfo.image.hsv_origins[HSVIndex::S], 0, 0, mask);
	imginfo.image.hsv_origins[HSVIndex::S].setTo(1, mask);
	cv::inRange(imginfo.image.hsv_origins[HSVIndex::V], 0, 0, mask);
	imginfo.image.hsv_origins[HSVIndex::V].setTo(1, mask);

	/* init diff matrix */
	imginfo.filter.bgr_filters.resize(3);
	imginfo.filter.hls_filters.resize(3);
	imginfo.filter.hsv_filters.resize(3);

	imginfo.filter.bgr_filters[BGRIndex::B] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.bgr_filters[BGRIndex::G] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.bgr_filters[BGRIndex::R] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);

	imginfo.filter.hls_filters[HLSIndex::H] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.hls_filters[HLSIndex::L] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.hls_filters[HLSIndex::S] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);

	imginfo.filter.hsv_filters[HSVIndex::H] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.hsv_filters[HSVIndex::S] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);
	imginfo.filter.hsv_filters[HSVIndex::V] = Mat::zeros(imginfo.row, imginfo.col, CV_16S);

	imginfo.filter.diff = Mat::zeros(imginfo.row, imginfo.col, CV_16S);

	//*******************************************************************************************************

	// Gamma
	imginfo.filter.hsv_filters[HSVIndex::V].convertTo(imginfo.filter.gamma_mask, CV_32F);
	cv::multiply(1. / 255, imginfo.filter.gamma_mask, imginfo.filter.gamma_mask);

	//Clarity
	cv::bilateralFilter(imginfo.image.bgr, imginfo.filter.clarity_filter, DISTANCE, SIGMA_COLOR, SIGMA_SPACE);
	imginfo.filter.clarity_mask_U = Mat::zeros(imginfo.row, imginfo.col, CV_8UC3);
	imginfo.filter.clarity_mask_S = Mat::zeros(imginfo.row, imginfo.col, CV_16SC3);


	//Vignette
	Mat kernel_x,kernel_x_transpose, kernel_y, kernel_res;
	kernel_x = cv::getGaussianKernel(imginfo.col, 1000,CV_32F);
	kernel_y = cv::getGaussianKernel(imginfo.row, 1000,CV_32F);
	cv::transpose(kernel_x, kernel_x_transpose);
	kernel_res = (kernel_y * kernel_x_transpose);
	cv::normalize(kernel_res, kernel_res, 0,1, NORM_MINMAX);
	cv::subtract(1,kernel_res,kernel_res);
	// cout<<kernel_res<<endl;
	kernel_res = cv::abs(kernel_res);
	// kernel_res.convertTo(kernel_res,CV_32F);
	// imginfoi.mage.hsv_origins[HSVIndex::V].convertTo(imginfo.image.hsv_origins[HSVIndex::V],CV_32F);
	// cv::multiply(kernel_res,100,kernel_res,0.01,CV_32F);
	// imginfo.image.hsv_origins[HSVIndex::V].convertTo(imginfo.image.hsv_origins[HSVIndex::V],CV_8U);
	cv::multiply(125,kernel_res,kernel_res);
	kernel_res.convertTo(kernel_res,CV_16S);
	imginfo.filter.gaussian_kernel = kernel_res.clone();				//getUMat(cv::ACCESS_RW);
	// cout<<kernel_res<<endl;
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
	cv::imshow(TEST_WINDOW, imginfo.image.bgr);

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
	// cv::createTrackbar("Value", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_value);
	cv::setTrackbarPos("Value", SET_WINDOW, TRACKBAR_MID);

	// Temperature
	// cv::createTrackbar("Vibrance", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_vibrance);
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
