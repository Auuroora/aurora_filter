#include "define.h"
#include "header.h"

using namespace cv;
using namespace std;

double calculate_gaussian_normal_distribution(double x, double w, double std, double mu = 0)
{
	return w * pow(EXP, -((x - mu) * (x - mu)) / (2 * std * std)) / sqrt(2 * PI * std * std);
}

double make_weight_per_color(int color, int val)
{
	if (color == RED && val > 160)
		val -= 180;

	switch (color)
	{
	case RED:
		if (val < 0)	
			return calculate_gaussian_normal_distribution(abs(color - val), 80.0, 9.0);
		else
			return calculate_gaussian_normal_distribution(abs(color - val), 40.0, 4.5);

	case ORANGE:
		return calculate_gaussian_normal_distribution(abs(color - val), 45.0, 7.0);
	case YELLOW:
		return calculate_gaussian_normal_distribution(abs(color - val), 40.0, 7.5);
	case GREEN:
		return calculate_gaussian_normal_distribution(abs(color - val), 120, 14);
	case BLUE:
		return calculate_gaussian_normal_distribution(abs(color - val), 120, 14);
	case VIOLET:
		return calculate_gaussian_normal_distribution(abs(color - val), 110, 12);
	}
	return 0;
}

double make_weight_per_saturation(int val, int mu)
{
	return calculate_gaussian_normal_distribution((double)val, 300.0, 100.0, (double)mu);
}

double make_weight_per_value(int val, int mu)
{
	return calculate_gaussian_normal_distribution((double)val, 300.0, 100.0, (double)mu);
}

void downsize_img(Mat &src, Mat &dst, int &downsizedRow, int &downsizedCol)
{
	int row = src.rows;
	int col = src.cols;
	// TO DO
}

/*****************************************************************************
*							apply_filter
*	add bgr filter -> convert to hsv -> add hsv filter -> convert to bgr(res)
********************************************************************************/

void apply_filter()
{
	// apply BGR
	imginfo.bgr_img.convertTo(imginfo.bgr_img, CV_16SC3);
	cv::merge(imginfo.filter.bgr_filters, imginfo.filter.bgr_filter);
	cv::add(imginfo.bgr_img, imginfo.filter.bgr_filter, imginfo.res_img);
	imginfo.res_img.convertTo(imginfo.res_img, CV_8UC3);

	// apply HSV
	cv::cvtColor(imginfo.res_img, imginfo.res_img, COLOR_BGR2HSV);
	imginfo.res_img.convertTo(imginfo.res_img, CV_16SC3);
	cv::merge(imginfo.filter.hsv_filters, imginfo.filter.hsv_filter);
	cv::add(imginfo.res_img, imginfo.filter.hsv_filter, imginfo.res_img);
	imginfo.res_img.convertTo(imginfo.res_img, CV_8UC3);

	cv::cvtColor(imginfo.res_img, imginfo.res_img, COLOR_HSV2BGR);
}

void update_hue(int pos)
{
	//Mat diff = Mat::ones(imginfo.originImg.rows, imginfo.originImg.cols, CV_16S) * (pos - imginfo.trackbar.color.hue);
	
	//cv::add(imginfo.filter.hue, diff, imginfo.filter.hue);
	//imginfo.trackbar.color.hue = pos;

	//cv::parallel_for_(Range(0, imginfo.originImg.rows * imginfo.originImg.cols), ParallelModulo(splitImg[H], splitImg[H], (HUE_MAX + 1)));
}

void update_saturation(int pos)
{
	//imginfo.filter.diff.setTo(pos - imginfo.trackbar.color.sat);
	//cv::add(imginfo.filter.sat, imginfo.filter.diff, imginfo.filter.sat);
	//imginfo.trackbar.color.sat = pos;
}

void update_value(int pos) 
{
	imginfo.filter.diff.setTo(pos - imginfo.trackbar.value);
	cv::add(imginfo.filter.hsv_filters[ColorSpaceIndex::V], imginfo.filter.diff, imginfo.filter.hsv_filters[ColorSpaceIndex::V]);
	imginfo.trackbar.value = pos;
}

void update_temperature(int pos)
{
	imginfo.filter.diff.setTo(abs(imginfo.trackbar.temperature));
	if (imginfo.trackbar.temperature >= 0)	
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::R],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	else
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);

	imginfo.filter.diff.setTo(abs(pos));
	if (pos >= 0)
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	else
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);

	imginfo.trackbar.temperature = pos;
}

void update_vibrance(int pos) {
   // 기존 값 빼고
   imginfo.filter.diff.setTo(imginfo.trackbar.vibrance);
   cv::subtract(
      imginfo.filter.hsv_filters[ColorSpaceIndex::S], 
      imginfo.filter.diff, 
      imginfo.filter.hsv_filters[ColorSpaceIndex::S]
   );

   // 변경치 * 가중치 더해서
   imginfo.filter.diff.setTo(pos);
   imginfo.filter.diff.convertTo(imginfo.filter.diff, CV_32F);
   imginfo.filter.diff = imginfo.filter.diff.mul(1);
   imginfo.filter.diff.convertTo(imginfo.filter.diff, CV_16S);

   // 적용
   cv::add(
      imginfo.filter.hsv_filters[ColorSpaceIndex::S], 
      imginfo.filter.diff, 
      imginfo.filter.hsv_filters[ColorSpaceIndex::S]
   );

   // 변경치 업데이트
   imginfo.trackbar.vibrance = pos;
}

void update_highlight_hue()
{
	//Mat add = Mat::ones(imginfo.originImg.rows, imginfo.originImg.cols, CV_32F) * (double)imginfo.trackbar.splittone.highlight;

	//add = add.mul(imginfo.weightMatrixPerValue);
	//add.convertTo(add, CV_8U);

	//cv::add(imginfo.originHsvSplit[H], add, imginfo.filterHsvSplit[H]);
}

void update_highlight_saturation()
{
	//Mat add = Mat::ones(imginfo.originImg.rows, imginfo.originImg.cols, CV_32F) * (double)imginfo.trackbar.splittone.highlight;

	//add = add.mul(imginfo.weightMatrixPerValue);
	//add.convertTo(add, CV_8U);

	//cv::add(imginfo.originHsvSplit[S], add, imginfo.filterHsvSplit[S]);
}

/*********************************************************************
*	이하 동훈이 코드 
**************************************************************/

void update_tint(int pos)
{
	imginfo.filter.diff.setTo(pos - imginfo.trackbar.tint);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	imginfo.trackbar.tint = pos;
}

void update_clarity(int pos)
{
	float clarityValue_f = pos / (float)10.0;
	cv::addWeighted(imginfo.bgr_img, clarityValue_f, imginfo.filter.clarity_filter, -clarityValue_f, 0, imginfo.filter.clarity_mask);
	cv::add(imginfo.res_img, imginfo.filter.clarity_mask, imginfo.res_img);
}

// Refactoring : a,b 구하는건 함수로
void update_brightness_and_constrast(int brightnessValue, int constrastValue)
{
	double a,b;

	if(imginfo.trackbar.constrast>0){
		double delta = 127.*imginfo.trackbar.constrast/255.;
		a = 255./(255.-2*delta);
		b= a*(imginfo.trackbar.brightness-delta);
	}
	else
	{
		double delta = 127.*imginfo.trackbar.constrast/255.;
		a = (255.-2*delta)/255.;
		b = a*imginfo.trackbar.brightness+delta;
	}
	
	cv::multiply(imginfo.bgr_img,a-1,imginfo.filter.diff);
	cv::subtract(imginfo.res_img,imginfo.filter.diff,imginfo.res_img);
	imginfo.filter.diff.setTo(b);
	cv::subtract(imginfo.res_img,imginfo.filter.diff,imginfo.res_img);


	if(constrastValue>0){
		double delta = 127.*constrastValue/255.;
		a = 255./(255.-2*delta);
		b= a*(brightnessValue-delta);
	}
	else
	{
		double delta = 127.*constrastValue/255.;
		a = (255.-2*delta)/255.;
		b = a*brightnessValue+delta;
	}

	cv::multiply(imginfo.bgr_img,a-1,imginfo.filter.diff);
	cv::add(imginfo.res_img,imginfo.filter.diff,imginfo.res_img);
	imginfo.filter.diff.setTo(b);
	cv::add(imginfo.res_img,imginfo.filter.diff,imginfo.res_img);


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

	//tempImg.convertTo(temp1Img, CV_8U, a, b);

	//tempImg = temp1Img;
}

void update_exposure(int pos)
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

void update_vignette(int pos)			// 코드 옮기면서 변경함 -> IMG.depth 에서 에러 발생 가능
{  
	imginfo.filter.diff = imginfo.filter.gaussian_kernel.clone();
	cv::multiply(imginfo.filter.diff,imginfo.trackbar.vignette,imginfo.filter.diff);

	//양이 밝게 , 음이 어둡게
	if(imginfo.trackbar.vignette>0){
		cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::B],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::G],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::R],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	else if(imginfo.trackbar.vignette<0){
		cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::B],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::G],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::R],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	cv::divide(imginfo.filter.diff,imginfo.trackbar.vignette,imginfo.filter.diff);
	cv::multiply(imginfo.filter.diff,pos,imginfo.filter.diff);

	if(pos>0){
		cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::B],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::G],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::R],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	else if(pos<0){
		cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::B],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::G],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::R],imginfo.filter.diff,imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}

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