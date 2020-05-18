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

void update_vibrance(int pos)
{
	// 기존 값 빼고
	imginfo.filter.diff.setTo(imginfo.trackbar.vibrance);
	cv::subtract(
		imginfo.filter.hsv_filters[ColorSpaceIndex::S],
		imginfo.filter.diff,
		imginfo.filter.hsv_filters[ColorSpaceIndex::S]);

	// 변경치 * 가중치 더해서
	imginfo.filter.diff.setTo(pos);
	imginfo.filter.diff.convertTo(imginfo.filter.diff, CV_32F);
	imginfo.filter.diff = imginfo.filter.diff.mul(1);
	imginfo.filter.diff.convertTo(imginfo.filter.diff, CV_16S);

	// 적용
	cv::add(
		imginfo.filter.hsv_filters[ColorSpaceIndex::S],
		imginfo.filter.diff,
		imginfo.filter.hsv_filters[ColorSpaceIndex::S]);

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
	imginfo.filter.diff.setTo((pos - imginfo.trackbar.tint)/5.0);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	imginfo.trackbar.tint = pos;
}

void update_clarity(int pos)
{	
	double clarity_value;
	clarity_value = imginfo.trackbar.clarity/(double)10.0;
	cout<<clarity_value<<endl;

	cv::addWeighted(imginfo.downsized_img,clarity_value,imginfo.filter.clarity_filter,-clarity_value,0,imginfo.filter.clarity_mask_U);
	// cout<<imginfo.filter.clarity_mask_U.type()<<endl;
	cv::split(imginfo.filter.clarity_mask_S,imginfo.filter.clarity_mask_split);
	cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.clarity_mask_split[ColorSpaceIndex::B], imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.clarity_mask_split[ColorSpaceIndex::G], imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.clarity_mask_split[ColorSpaceIndex::R], imginfo.filter.bgr_filters[ColorSpaceIndex::R]);


	clarity_value = pos/(double)10.0;

	cv::addWeighted(imginfo.downsized_img,clarity_value,imginfo.filter.clarity_filter,-clarity_value,0,imginfo.filter.clarity_mask_U);
	// cout<<imginfo.filter.clarity_mask_U.type()<<endl;
	imginfo.filter.clarity_mask_U.convertTo(imginfo.filter.clarity_mask_S,CV_16SC3,0.8);
	cv::split(imginfo.filter.clarity_mask_S,imginfo.filter.clarity_mask_split);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.clarity_mask_split[ColorSpaceIndex::B], imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.clarity_mask_split[ColorSpaceIndex::G], imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.clarity_mask_split[ColorSpaceIndex::R], imginfo.filter.bgr_filters[ColorSpaceIndex::R]);

	imginfo.trackbar.clarity=pos;
}

// Refactoring : a,b 구하는건 함수로
void update_brightness_and_constrast(int brightness_pos, int constrast_pos)
{
	double a, b;
	cout<<brightness_pos<<"&"<<constrast_pos<<endl;
	if (imginfo.trackbar.constrast > 0)
	{
		double delta = MAX_7B_F * imginfo.trackbar.constrast / MAX_8B_F;
		a = MAX_8B_F / (MAX_8B_F - delta * 2);
		b = a * (imginfo.trackbar.brightness - delta);
	}
	else
	{
		double delta = -MAX_7B_F  * imginfo.trackbar.constrast / MAX_8B_F;
		a = (MAX_8B_F- delta * 2) / MAX_8B_F;
		b = a * imginfo.trackbar.brightness + delta;
	}
	cout<<a<<" "<<b<<endl;

	cv::multiply(imginfo.bgr_split[ColorSpaceIndex::B], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);


	cv::multiply(imginfo.bgr_split[ColorSpaceIndex::G], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);

	cv::multiply(imginfo.bgr_split[ColorSpaceIndex::R], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);

	imginfo.filter.diff.setTo(b);
	cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);

	if (constrast_pos > 0)
	{
		double delta = MAX_7B_F * constrast_pos / MAX_8B_F;
		a = MAX_8B_F / (MAX_8B_F - delta * 2);
		b = a * (brightness_pos - delta);
	}
	else
	{
		double delta = -MAX_7B_F  * constrast_pos / MAX_8B_F;
		a = (MAX_8B_F- delta * 2) / MAX_8B_F;
		b = a * brightness_pos + delta;
	}

	cv::multiply(imginfo.bgr_split[ColorSpaceIndex::B], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);


	cv::multiply(imginfo.bgr_split[ColorSpaceIndex::G], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);

	cv::multiply(imginfo.bgr_split[ColorSpaceIndex::R], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);

	imginfo.filter.diff.setTo(b);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);


/////////////////////////////////////////////////////////////////////////////
	//brightness_pos -= BRIGHTNESS_MID;
	//constrast_pos -= CONSTRAST_MID;
	//tempImg = originImg.clone();

	//double a, b;

	//if (constrast_pos > 0)
	//{
	//	double delta = MAX_7B_F * constrast_pos / MAX_8B_F;
	//	a = MAX_8B_F / (MAX_8B_F - delta * 2);
	//	b = a * (brightness_pos - delta);
	//}
	//else
	//{
	//	double delta = -MAX_7B_F * constrast_pos / MAX_8B_F;
	//	a = (MAX_8B_F - delta * 2) / MAX_8B_F;
	//	b = a * brightness_pos + delta;
	//}

	//tempImg.convertTo(temp1Img, CV_8U, a, b);

	//tempImg = temp1Img;
}

void update_exposure(int pos)
{
	imginfo.filter.diff.setTo(abs(imginfo.trackbar.exposure));

	if (imginfo.trackbar.exposure >= 0)
	{
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	else
	{
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}

	imginfo.filter.diff.setTo(abs(pos));
	if (pos >= 0)
	{
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::add(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	else
	{
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::subtract(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	imginfo.trackbar.exposure = pos;
}


void update_gamma(int pos)
{
	// double gammaValue=gamma/100.0;
	// double inv_gamma=1/gammaValue;

	cv::pow(imginfo.filter.diff, -imginfo.trackbar.gamma, imginfo.filter.diff);
	cv::multiply(255, imginfo.filter.diff, imginfo.filter.diff);
	cv::cvtColor(imginfo.filter.diff, imginfo.filter.diff, CV_8U);
	cv::subtract(imginfo.filter.hsv_filters[ColorSpaceIndex::V], imginfo.filter.diff, imginfo.filter.hsv_filters[ColorSpaceIndex::V]);

	imginfo.filter.diff = imginfo.filter.gamma_mask.clone();

	cv::pow(imginfo.filter.diff, pos, imginfo.filter.diff);
	cv::multiply(255, imginfo.filter.diff, imginfo.filter.diff);
	cv::cvtColor(imginfo.filter.diff, imginfo.filter.diff, CV_8U);
	cv::add(imginfo.filter.hsv_filters[ColorSpaceIndex::V], imginfo.filter.diff, imginfo.filter.hsv_filters[ColorSpaceIndex::V]);

	cv::cvtColor(imginfo.filter.diff, imginfo.filter.diff, CV_16S);
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
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_32F);
	imginfo.filter.diff = imginfo.filter.grain_mask.clone();

	cv::multiply(imginfo.filter.diff, (pos - imginfo.trackbar.grain)/3.0, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::add(imginfo.filter.hsv_filters[ColorSpaceIndex::V], imginfo.filter.diff, imginfo.filter.hsv_filters[ColorSpaceIndex::V]);

	imginfo.trackbar.grain = pos;

	//cv::add(imginfo.filter.grain_mask * grainValue / 10.0, hsv_Split[2], hsv_Split[2]);
	//hsv_Split[2].convertTo(hsv_Split[2], CV_8U);
	//cv::multiply(2.0, hsv_Split[2], hsv_Split[2]);

	//merge(hsv_Split, 3, hsvImg);
	//cv::cvtColor(hsvImg, tempImg, COLOR_HSV2BGR);
}

void update_vignette(int pos) // 코드 옮기면서 변경함 -> IMG.depth 에서 에러 발생 가능
{
	cv::split(imginfo.res_img,imginfo.res_split);
	imginfo.filter.diff = imginfo.filter.gaussian_kernel.clone();
	imginfo.res_split[ColorSpaceIndex::B].convertTo(imginfo.res_split[ColorSpaceIndex::B],CV_32F);
	imginfo.res_split[ColorSpaceIndex::R].convertTo(imginfo.res_split[ColorSpaceIndex::R],CV_32F);
	imginfo.res_split[ColorSpaceIndex::G].convertTo(imginfo.res_split[ColorSpaceIndex::G],CV_32F);

	//양이 밝게 , 음이 어둡게
	// if (imginfo.trackbar.vignette > 0)
	// {
	// 	cv::add(imginfo.res_split[ColorSpaceIndex::B], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	// 	cv::add(imginfo.res_split[ColorSpaceIndex::G], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	// 	cv::add(imginfo.res_split[ColorSpaceIndex::R], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	// }
	// else if (imginfo.trackbar.vignette < 0)
	// {
	// 	cv::subtract(imginfo.res_split[ColorSpaceIndex::B], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	// 	cv::subtract(imginfo.res_split[ColorSpaceIndex::G], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	// 	cv::subtract(imginfo.res_split[ColorSpaceIndex::R], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	// }

	if (pos > 0)
	{
		cv::subtract(imginfo.res_split[ColorSpaceIndex::B], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::subtract(imginfo.res_split[ColorSpaceIndex::G], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::subtract(imginfo.res_split[ColorSpaceIndex::R], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}
	else if (pos < 0)
	{
		cv::add(imginfo.res_split[ColorSpaceIndex::B], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
		cv::add(imginfo.res_split[ColorSpaceIndex::G], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
		cv::add(imginfo.res_split[ColorSpaceIndex::R], imginfo.filter.diff*100, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	}


	// imginfo.filter.bgr_filters[ColorSpaceIndex::B].convertTo(imginfo.filter.bgr_filters[ColorSpaceIndex::B],CV_8U);
	// imginfo.filter.bgr_filters[ColorSpaceIndex::R].convertTo(imginfo.filter.bgr_filters[ColorSpaceIndex::R],CV_8U);
	// imginfo.filter.bgr_filters[ColorSpaceIndex::G].convertTo(imginfo.filter.bgr_filters[ColorSpaceIndex::G],CV_8U);
	cv::convertScaleAbs(imginfo.filter.bgr_filters[ColorSpaceIndex::B],imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	cv::convertScaleAbs(imginfo.filter.bgr_filters[ColorSpaceIndex::G],imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	cv::convertScaleAbs(imginfo.filter.bgr_filters[ColorSpaceIndex::R],imginfo.filter.bgr_filters[ColorSpaceIndex::R]);

	
	cv::merge(imginfo.filter.bgr_filters,imginfo.res_img);

	
/*
	cv::split(imginfo.res_img,imginfo.res_split);
	imginfo.filter.diff = imginfo.filter.gaussian_kernel.clone();
	imginfo.res_split[ColorSpaceIndex::B].convertTo(imginfo.res_split[ColorSpaceIndex::B],CV_64F);
	imginfo.res_split[ColorSpaceIndex::R].convertTo(imginfo.res_split[ColorSpaceIndex::R],CV_64F);
	imginfo.res_split[ColorSpaceIndex::G].convertTo(imginfo.res_split[ColorSpaceIndex::G],CV_64F);

	//양이 밝게 , 음이 어둡게
	if (imginfo.trackbar.vignette > 0)
	{
		cv::multiply(imginfo.res_split[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::B]);
		cv::multiply(imginfo.res_split[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::G]);
		cv::multiply(imginfo.res_split[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::R]);
	}
	else if (imginfo.trackbar.vignette < 0)
	{
		cv::divide(imginfo.res_split[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::B]);
		cv::divide(imginfo.res_split[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::G]);
		cv::divide(imginfo.res_split[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::R]);
	}

	if (pos > 0)
	{
		cv::multiply(imginfo.res_split[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::B]);
		cv::multiply(imginfo.res_split[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::G]);
		cv::multiply(imginfo.res_split[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::R]);
	}
	else if (pos < 0)
	{
		cv::divide(imginfo.res_split[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::B]);
		cv::divide(imginfo.res_split[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::G]);
		cv::divide(imginfo.res_split[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.res_split[ColorSpaceIndex::R]);
	}


	// imginfo.res_split[ColorSpaceIndex::B].convertTo(imginfo.res_split[ColorSpaceIndex::B],CV_8U);
	// imginfo.res_split[ColorSpaceIndex::R].convertTo(imginfo.res_split[ColorSpaceIndex::R],CV_8U);
	// imginfo.res_split[ColorSpaceIndex::G].convertTo(imginfo.res_split[ColorSpaceIndex::G],CV_8U);
	cv::convertScaleAbs(imginfo.res_split[ColorSpaceIndex::B],imginfo.res_split[ColorSpaceIndex::B]);
	cv::convertScaleAbs(imginfo.res_split[ColorSpaceIndex::G],imginfo.res_split[ColorSpaceIndex::G]);
	cv::convertScaleAbs(imginfo.res_split[ColorSpaceIndex::R],imginfo.res_split[ColorSpaceIndex::R]);

	
	cv::merge(imginfo.res_split,imginfo.res_img);


*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// imginfo.filter.diff = imginfo.filter.gaussian_kernel.clone();
	// cv::multiply(imginfo.filter.diff, imginfo.trackbar.vignette, imginfo.filter.diff);
	// imginfo.filter.bgr_filters[ColorSpaceIndex::B].convertTo(imginfo.filter.bgr_filters[ColorSpaceIndex::B],CV_64F);
	// imginfo.filter.bgr_filters[ColorSpaceIndex::R].convertTo(imginfo.filter.bgr_filters[ColorSpaceIndex::R],CV_64F);
	// imginfo.filter.bgr_filters[ColorSpaceIndex::G].convertTo(imginfo.filter.bgr_filters[ColorSpaceIndex::G],CV_64F);

	// //양이 밝게 , 음이 어둡게
	// if (imginfo.trackbar.vignette > 0)
	// {
	// 	cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::B], .fimginfoilter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	// 	cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	// 	cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	// }
	// else if (imginfo.trackbar.vignette < 0)
	// {
	// 	cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	// 	cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	// 	cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	// }
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[ColorSpaceIndex::B],imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[ColorSpaceIndex::G],imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[ColorSpaceIndex::R],imginfo.filter.bgr_filters[ColorSpaceIndex::R]);


	// cv::divide(imginfo.filter.diff, imginfo.trackbar.vignette, imginfo.filter.diff);
	// cv::multiply(imginfo.filter.diff, pos, imginfo.filter.diff);
	// cout<<imginfo.filter.bgr_filters[ColorSpaceIndex::B].type()<<endl;
	// cout<<imginfo.filter.diff.type()<<endl;

	// if (pos >= 0)
	// {
	// 	cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	// 	cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	// 	cv::divide(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	// }
	// else if (pos < 0)
	// {
	// 	cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::B], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	// 	cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::G], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	// 	cv::multiply(imginfo.filter.bgr_filters[ColorSpaceIndex::R], imginfo.filter.diff, imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	// }
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[ColorSpaceIndex::B],imginfo.filter.bgr_filters[ColorSpaceIndex::B]);
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[ColorSpaceIndex::G],imginfo.filter.bgr_filters[ColorSpaceIndex::G]);
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[ColorSpaceIndex::R],imginfo.filter.bgr_filters[ColorSpaceIndex::R]);
	// cout<<"DSDDS"<<endl;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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