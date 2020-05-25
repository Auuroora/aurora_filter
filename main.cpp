#include "define.h"
#include "header.h"

WorkingImgInfo imginfo;

int main()
{
	/*********************************************************************
	*	OpenCL Test
	*********************************************************************/
	// OpenCL을 사용할 수 있는지 테스트
	if (!cv::ocl::haveOpenCL())
	{
		std::cout << "에러 : OpenCL을 사용할 수 없는 시스템입니다." << std::endl;
		return -1;
	}

	// 컨텍스트 생성
	cv::ocl::Context context;
	if (!context.create(cv::ocl::Device::TYPE_GPU))
	{
		std::cout << " 에러 : 컨텍스트를 생성할 수 없습니다." << std::endl;
		return -1;
	}

	// GPU 장치 정보
	std::cout << context.ndevices() << " GPU device (s) detected " << std::endl;
	for (size_t i = 0; i < context.ndevices(); i++)
	{
		cv::ocl::Device device = context.device(i);
		std::cout << " - Device " << i << " --- " << std::endl;
		std::cout << " Name : " << device.name() << std::endl;
		std::cout << " Availability : " << device.available() << std::endl;
		std::cout << "Image Support : " << device.imageSupport() << std::endl;
		std::cout << " OpenCL C version : " << device.OpenCL_C_Version() << std::endl;
	}

	cv::ocl::Device(context.device(0));
	cv::ocl::setUseOpenCL(true);

	/*********************************************************************
	*	Init
	*********************************************************************/
	//성수
	// cv::Mat inputImg = cv::imread("./img/test10.jpg", cv::IMREAD_COLOR);

	//동훈
	cv::Mat inputImg = cv::imread("2400x1600.jpg", cv::IMREAD_COLOR);
	if (inputImg.empty())
	{
		std::cout << "Image Open Failed" << std::endl;
		return -1;
	}

	imginfo.init_all(inputImg, 642, 432);
	/*********************************************************************
	*	Make Window
	*********************************************************************/
	cv::namedWindow(TEST_WINDOW, cv::WINDOW_NORMAL);
	cv::resizeWindow(TEST_WINDOW, 400, 600);
	cv::setMouseCallback(TEST_WINDOW, mouse_callback, &imginfo.image.res);

	cv::namedWindow(SET_WINDOW, cv::WINDOW_NORMAL);
	cv::resizeWindow(SET_WINDOW, 500, 200);
	cv::imshow(TEST_WINDOW, imginfo.image.bgr);

	cv::namedWindow("Origin", cv::WINDOW_NORMAL);
	cv::resizeWindow("Origin", 400, 600);
	cv::imshow("Origin", imginfo.image.bgr);

	/*********************************************************************
	*	Make Trackbar
	*********************************************************************/
	// HUE
	cv::createTrackbar("Hue", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_hue);
	cv::setTrackbarPos("Hue", SET_WINDOW, TRACKBAR_MID);

	// Saturation
	cv::createTrackbar("Sat", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_saturation);
	cv::setTrackbarPos("Sat", SET_WINDOW, TRACKBAR_MID);

	// lightness
	cv::createTrackbar("lightness", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_lightness);
	cv::setTrackbarPos("lightness", SET_WINDOW, TRACKBAR_MID);

	//Temperature
	cv::createTrackbar("temperature", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_temperature);
	cv::setTrackbarPos("temperature", SET_WINDOW, TRACKBAR_MID);

	// tint
	cv::createTrackbar("tint", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_tint);
	cv::setTrackbarPos("tint", SET_WINDOW, TRACKBAR_MID);

	// Vibrance
	cv::createTrackbar("vibrance", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_vibrance);
	cv::setTrackbarPos("vibrance", SET_WINDOW, TRACKBAR_MID);

	// Highlight
	cv::createTrackbar("hs", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_highlight_saturation);
	cv::setTrackbarPos("hs", SET_WINDOW, TRACKBAR_MID);
	cv::createTrackbar("hh", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_highlight_hue);
	cv::setTrackbarPos("hh", SET_WINDOW, TRACKBAR_MID);

	// Shadow
	cv::createTrackbar("ss", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_shadow_saturation);
	cv::setTrackbarPos("ss", SET_WINDOW, TRACKBAR_MID);
	cv::createTrackbar("sh", SET_WINDOW, TRACKBAR_MIN, TRACKBAR_MAX, on_change_shadow_hue);
	cv::setTrackbarPos("sh", SET_WINDOW, TRACKBAR_MID);

	while (cv::waitKey(0) != 27)
		;
	cv::destroyAllWindows();
	return 0;
}