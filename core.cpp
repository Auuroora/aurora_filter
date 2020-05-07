#include "define.h"
#include "header.h"

using namespace cv;
using namespace std;

double GND(double x, double w, double std, double mu = 0)
{
	return w * pow(EXP, -((x - mu) * (x - mu)) / (2 * std * std)) / sqrt(2 * PI * std * std);
}

double weightPerColor(int color, int val)
{
	if (color == RED && val > 160)
		val -= 180;

	switch (color)
	{
	case RED:
		if (val < 0)
			return GND(abs(color - val), 80.0, 9.0);
		else
			return GND(abs(color - val), 40.0, 4.5);

	case ORANGE:
		return GND(abs(color - val), 45.0, 7.0);
	case YELLOW:
		return GND(abs(color - val), 40.0, 7.5);
	case GREEN:
		return GND(abs(color - val), 120, 14);
	case BLUE:
		return GND(abs(color - val), 120, 14);
	case VIOLET:
		return GND(abs(color - val), 110, 12);
	}
	return 0;
}

double weightPerSaturation(int val, int mu)
{
	return GND((double)val, 300.0, 100.0, (double)mu);
}

double weightPerValue(int val, int mu)
{
	return GND((double)val, 300.0, 100.0, (double)mu);
}

void downsizing(Mat &src, Mat &dst, int &downsizedRow, int &downsizedCol)
{
	int row = src.rows;
	int col = src.cols;
	// TO DO
}

/*****************************************************************************
*							applyFilter
*	add bgr filter -> convert to hsv -> add hsv filter -> convert to bgr(res)
*****************************************************************************/
void applyFilter()
{
	// apply BGR
	imginfo.bgrImg.convertTo(imginfo.bgrImg, CV_16SC3);
	cv::merge(imginfo.filter.bgr_filters, imginfo.filter.bgr_filter);
	cv::add(imginfo.bgrImg, imginfo.filter.bgr_filter, imginfo.resImg);
	imginfo.resImg.convertTo(imginfo.resImg, CV_8UC3);

	// apply HSV
	cv::cvtColor(imginfo.resImg, imginfo.resImg, COLOR_BGR2HSV);
	imginfo.resImg.convertTo(imginfo.resImg, CV_16SC3);
	cv::merge(imginfo.filter.hsv_filters, imginfo.filter.hsv_filter);
	cv::add(imginfo.resImg, imginfo.filter.hsv_filter, imginfo.resImg);
	imginfo.resImg.convertTo(imginfo.resImg, CV_8UC3);

	cv::cvtColor(imginfo.resImg, imginfo.resImg, COLOR_HSV2BGR);
}

//void applyFilter() {
//	Mat bgr_filters[] = {
//		imginfo.filter.blue,
//		imginfo.filter.green,
//		imginfo.filter.red
//	};
//	Mat hsv_filters[] = {
//		imginfo.filter.hue,
//		imginfo.filter.sat,
//		imginfo.filter.val
//	};
//	Mat hsv_filter, bgr_filter;
//
//	// apply BGR
//	imginfo.bgrImg.convertTo(imginfo.bgrImg, CV_16SC3);
//	cv::merge(bgr_filters, 3, bgr_filter);
//	cv::add(imginfo.bgrImg, bgr_filter, imginfo.resImg);
//	imginfo.resImg.convertTo(imginfo.resImg, CV_8UC3);
//
//	// apply HSV
//	cv:cvtColor(imginfo.resImg, imginfo.resImg, COLOR_BGR2HSV);
//	imginfo.resImg.convertTo(imginfo.resImg, CV_16SC3);
//	cv::merge(hsv_filters, 3, hsv_filter);
//	cv::add(imginfo.resImg, hsv_filter, imginfo.resImg);
//	imginfo.resImg.convertTo(imginfo.resImg, CV_8UC3);
//
//	cv::cvtColor(imginfo.resImg, imginfo.resImg, COLOR_HSV2BGR);
//}

void updateHue(int pos)
{
	//Mat diff = Mat::ones(imginfo.originImg.rows, imginfo.originImg.cols, CV_16S) * (pos - imginfo.trackbar.color.hue);
	//
	//cv::add(imginfo.filter.hue, diff, imginfo.filter.hue);
	//imginfo.trackbar.color.hue = pos;

	//cv::parallel_for_(Range(0, imginfo.originImg.rows * imginfo.originImg.cols), ParallelModulo(splitImg[H], splitImg[H], (HUE_MAX + 1)));
}

void updateSaturation(int pos)
{
	//imginfo.filter.diff.setTo(pos - imginfo.trackbar.color.sat);
	//cv::add(imginfo.filter.sat, imginfo.filter.diff, imginfo.filter.sat);
	//imginfo.trackbar.color.sat = pos;
}

void updateValue(int pos)
{
	//imginfo.filter.diff.setTo(pos - imginfo.trackbar.color.val);
	//cv::add(imginfo.filter.val, imginfo.filter.diff, imginfo.filter.val);
	//imginfo.trackbar.color.val = pos;
}

void updateTemperature(int pos)
{
	imginfo.filter.diff.setTo(abs(imginfo.trackbar.temperature));
	if (imginfo.trackbar.temperature >= 0)
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	else
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);

	imginfo.filter.diff.setTo(abs(pos));
	if (pos >= 0)
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	else
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);

	imginfo.trackbar.temperature = pos;
}

void updateVibrance()
{
	//int len = imginfo.originImg.rows * imginfo.originImg.cols;
	//ParallelChangeVibrance parallelChangeVibrance(imginfo.originHsvSplit[S], imginfo.filterHsvSplit[S]);
	//cv::parallel_for_(Range(0, len), parallelChangeVibrance);
}

void updateHighlightHue()
{
	//Mat add = Mat::ones(imginfo.originImg.rows, imginfo.originImg.cols, CV_32F) * (double)imginfo.trackbar.splittone.highlight;

	//add = add.mul(imginfo.weightMatrixPerValue);
	//add.convertTo(add, CV_8U);

	//cv::add(imginfo.originHsvSplit[H], add, imginfo.filterHsvSplit[H]);
}

void updateHighlightSaturation()
{
	//Mat add = Mat::ones(imginfo.originImg.rows, imginfo.originImg.cols, CV_32F) * (double)imginfo.trackbar.splittone.highlight;

	//add = add.mul(imginfo.weightMatrixPerValue);
	//add.convertTo(add, CV_8U);

	//cv::add(imginfo.originHsvSplit[S], add, imginfo.filterHsvSplit[S]);
}

/*********************************************************************
*	이하 동훈이 코드
*********************************************************************/
void update_tint(int pos)
{
	imginfo.filter.diff.setTo(pos - imginfo.trackbar.tint);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	imginfo.trackbar.tint = pos;
}

void update_clarity(int pos)
{
	float clarityValue_f = pos / (float)10.0;
	cv::addWeighted(imginfo.bgrImg, clarityValue_f, imginfo.filter.clarity_filter, -clarityValue_f, 0, imginfo.filter.clarity_mask);
	cv::add(imginfo.resImg, imginfo.filter.clarity_mask, imginfo.resImg);
}

void update_brightness(int pos)
{
}

void update_constrast(int brightnessValue, int constrastValue)
{
	//brightnessValue -= BRIGHTNESS_MID;
	//constrastValue -= CONSTRAST_MID;
	//tempImg = originImg.clone();

	//double a, b;

	//if (constrastValue > 0)
	//{
	//	double delta = MAX_7B_F * constrastValue / MAX_8B_F;
	//	a = MAX_8B_F / (MAX_8B_F - delta * 2);
	//	b = a * (brightnessValue - delta);
	//}
	//else
	//{
	//	double delta = -MAX_7B_F * constrastValue / MAX_8B_F;
	//	a = (MAX_8B_F - delta * 2) / MAX_8B_F;
	//	b = a * brightnessValue + delta;
	//}

	////굳이 이걸 마스크로 표현해야하나?
	//tempImg.convertTo(temp1Img, CV_8U, a, b);
	//tempImg = temp1Img;
}

void upadate_exposure(int pos)
{
	////메모리를 아끼냐 성능을 아끼냐 차이로 추후 업뎃
	imginfo.filter.diff.setTo(2*abs(imginfo.trackbar.exposure));

	if (imginfo.trackbar.exposure >= 0) {
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	else {
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}

	imginfo.filter.diff.setTo(2*abs(pos));
	if(pos>=0){
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	else {
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	imginfo.trackbar.exposure = pos;
}

void update_gamma(int pos)
{
	//double gammaValue=gamma/100.0;
	//double inv_gamma=1/gammaValue;

	imginfo.filter.diff=imginfo.filter.gamma_mask.clone();
	cv::cvtColor(imginfo.filter.diff,imginfo.filter.diff,CV_32F);


	cv::pow(imginfo.filter.diff,-imginfo.trackbar.gamma,imginfo.filter.diff);
	cv::multiply(255,imginfo.filter.diff,imginfo.filter.diff);
	cv::cvtColor(imginfo.filter.diff,imginfo.filter.diff,CV_8U);
	cv::subtract(imginfo.filter.hsv_filters[ColorSpaceIndex::V],imginfo.filter.diff,imginfo.filter.hsv_filters[ColorSpaceIndex::V]);

	imginfo.filter.diff=imginfo.filter.gamma_mask.clone();

	cv::pow(imginfo.filter.diff,pos,imginfo.filter.diff);
	cv::multiply(255,imginfo.filter.diff,imginfo.filter.diff);
	cv::cvtColor(imginfo.filter.diff,imginfo.filter.diff,CV_8U);
	cv::add(imginfo.filter.hsv_filters[ColorSpaceIndex::V],imginfo.filter.diff,imginfo.filter.hsv_filters[ColorSpaceIndex::V]);


	cv::cvtColor(imginfo.filter.diff,imginfo.filter.diff,CV_16S);
	imginfo.trackbar.gamma = pos;

	// hsv_Split[2].convertTo(hsv_Split[2],CV_32F);
	
	// cv::multiply(1./255,hsv_Split[2],hsv_Split[2]);
	// cv::pow(hsv_Split[2],inv_gamma,hsv_Split[2]);
	// cv::multiply(255,hsv_Split[2],hsv_Split[2]);
	// hsv_Split[2].convertTo(hsv_Split[2],CV_8U);
	// cv::merge(hsv_Split,3,hsvImg);
	// cv::cvtColor(hsvImg,newImg,COLOR_HSV2BGR);
	// cv::imshow("Gamma",newImg);
}

//
void update_grain(int pos)
{
	imginfo.filter.diff = imginfo.filter.grain_mask.clone();


	cv::multiply(imginfo.filter.diff,(pos-imginfo.trackbar.grain),imginfo.filter.diff);
	cv::add(imginfo.filter.hsv_filters[ColorSpaceIndex::V],imginfo.filter.diff,imginfo.filter.hsv_filters[ColorSpaceIndex::V]);
	
	imginfo.trackbar.grain=pos;
	
	//cv::add(imginfo.filter.grain_mask * grainValue / 10.0, hsv_Split[2], hsv_Split[2]);
	//hsv_Split[2].convertTo(hsv_Split[2], CV_8U);
	//cv::multiply(2.0, hsv_Split[2], hsv_Split[2]);

	//merge(hsv_Split, 3, hsvImg);
	//cv::cvtColor(hsvImg, tempImg, COLOR_HSV2BGR);
}

void update_vignette(int pos)
{
	//cv::split(tempImg, rgb_Split);

	///*
	//	cout.setf(ios::left);
	//	cout<<setw(20)<<"Name"<<setw(10)<<"Size"<<"\t\tDepth"<<endl;
	//	cout<<setw(20)<<"temp_img"<<tempImg.size<<"\t\t"<<tempImg.depth()<<endl;
	//	cout<<setw(20)<<"rgb_Split[i]"<<rgb_Split[0].size<<"\t\t"<<rgb_Split[0].depth()<<endl;
	//	cout<<setw(20)<<"kernel"<<kernel.size<<"\t\t"<<kernel.depth()<<endl;
	//	cout<<setw(20)<<"mask"<<mask.size<<"\t\t"<<mask.depth()<<endl;
	//*/

	//for (int i = 0; i < 3; i++) {
	//	rgb_Split[i].convertTo(processed_image[i], CV_64F);
	//	//cout<<setw(20)<<"process_img_i"<<processed_image[i].size<<"\t\t"<<processed_image[i].depth()<<endl;
	//	//어둡게
	//	//multiply(processed_image[i],mask,processed_image[i]);

	//	//밝게
	//	cv::divide(processed_image[i], mask, processed_image[i]);

	//	cv::convertScaleAbs(processed_image[i], processed_image[i]);
	//}
	//cv::merge(processed_image, 3, tempImg);
}