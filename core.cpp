#include "define.h"
#include "header.h"

using namespace cv;
using namespace std;

double calculate_gaussian_normal_distribution(double x, double w, double std, double mu = 0)
{
	return w * pow(EXP, -((x - mu) * (x - mu)) / (2 * std * std)) / sqrt(2 * PI * std * std);
}

double weight_per_color(int color, int val) {
	if (color == RED && val > 160) val -= 180;

	switch (color) {
	case RED:
		if (val < 0) return GND(abs(color - val), 80.0, 9.0);
		else return GND(abs(color - val), 40.0, 4.5);

	case ORANGE:	return GND(abs(color - val), 45.0, 7.0);
	case YELLOW:	return GND(abs(color - val), 40.0, 7.5);
	case GREEN:		return GND(abs(color - val), 120, 14);
	case BLUE:		return GND(abs(color - val), 120, 14);
	case VIOLET:	return GND(abs(color - val), 110, 12);
	}
	return 0;
}

double weight_per_saturation(int val, int mu) {
	return GND((double)val, 200.0, 50.0, (double)mu);
}

double weight_per_value(int val, int mu) {
	return GND((double)val, 200.0, 50.0, (double)mu);
}

void downsizing(Mat &src, Mat &dst, int downsizedRow, int downsizedCol) {
	if (src.rows >= downsizedRow && src.cols >= downsizedCol) {
		resize(src, dst, Size(downsizedRow, downsizedCol), 0, 0, INTER_LINEAR);
	}
	else {
		dst = src.clone();
	}
}

/*****************************************************************************
*							applyFilter
*	add bgr filter -> convert to hsv -> add hsv filter -> convert to bgr(res)
*****************************************************************************/
void apply_filter() {
	switch (imginfo.changed_color_space) {
	case BGR_CHANGED:
		imginfo.image.bgr.convertTo(imginfo.image.bgr, CV_16SC3);
		cv::merge(imginfo.filter.bgr_filters, imginfo.filter.bgr_filter);
		cv::add(imginfo.image.bgr, imginfo.filter.bgr_filter, imginfo.image.res);
		imginfo.image.res.convertTo(imginfo.image.res, CV_8UC3);
		break;

	case HLS_CHANGED:
		imginfo.image.hls.convertTo(imginfo.image.res, CV_16SC3);
		cv::merge(imginfo.filter.hls_filters, imginfo.filter.hls_filter);
		cv::add(imginfo.image.res, imginfo.filter.hls_filter, imginfo.image.res);
		imginfo.image.res.convertTo(imginfo.image.res, CV_8UC3);
		cv::cvtColor(imginfo.image.res, imginfo.image.res, COLOR_HLS2BGR);
		break;

	case HSV_CHANGED:
		imginfo.image.hsv.convertTo(imginfo.image.res, CV_16SC3);
		cv::merge(imginfo.filter.hsv_filters, imginfo.filter.hsv_filter);
		cv::add(imginfo.image.res, imginfo.filter.hsv_filter, imginfo.image.res);
		imginfo.image.res.convertTo(imginfo.image.res, CV_8UC3);
		cv::cvtColor(imginfo.image.res, imginfo.image.res, COLOR_HSV2BGR);
		break;
	}
}

void update_hue(int pos) {
	imginfo.filter.diff.setTo(pos - imginfo.trackbar.hue);
	cv::add(
		imginfo.filter.hls_filters[Cind::H],
		imginfo.filter.diff,
		imginfo.filter.hls_filters[Cind::H]
	);
	imginfo.trackbar.hue = pos;
	imginfo.changed_color_space = HLS_CHANGED;
}

void update_saturation(int pos) {
	imginfo.filter.diff.setTo(pos - imginfo.trackbar.saturation);
	cv::add(
		imginfo.filter.hls_filters[Cind::S],
		imginfo.filter.diff,
		imginfo.filter.hls_filters[Cind::S]
	);
	imginfo.trackbar.saturation = pos;
	imginfo.changed_color_space = HLS_CHANGED;
}

void update_lightness(int pos) {
	imginfo.filter.diff.setTo(pos - imginfo.trackbar.lightness);
	cv::add(
		imginfo.filter.hls_filters[Cind::L],
		imginfo.filter.diff,
		imginfo.filter.hls_filters[Cind::L]
	);
	imginfo.trackbar.lightness = pos;
	imginfo.changed_color_space = HLS_CHANGED;
}

void update_temperature(int pos) {
	imginfo.filter.diff.setTo(abs(imginfo.trackbar.temperature));
	if (imginfo.trackbar.temperature >= 0)
		cv::subtract(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
	else
		cv::subtract(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);

	imginfo.filter.diff.setTo(abs(pos));
	if (pos >= 0)
		cv::add(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
	else
		cv::add(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);

	imginfo.trackbar.temperature = pos;
	imginfo.changed_color_space = BGR_CHANGED;
}

void update_vibrance(int pos) {
	Mat mask;
	Mat tmp = imginfo.image.hsv_origins[Cind::Sat].clone();
	tmp.convertTo(tmp, CV_16S);

	cv::add(
		tmp,
		imginfo.filter.hsv_filters[Cind::Sat],
		tmp
	);

	cv::inRange(tmp, 0, 100, mask);
	cv::addWeighted(
		(100 / (imginfo.image.hsv_origins[Cind::Sat])),
		(pos - imginfo.trackbar.vibrance),
		0,
		0,
		0,
		imginfo.filter.diff,
		CV_16S
	);
	cv::add(
		imginfo.filter.hsv_filters[Cind::Sat],
		imginfo.filter.diff,
		imginfo.filter.hsv_filters[Cind::Sat],
		mask
	);

	//Mat tmp = imginfo.filter.hsv_filters[Cind::Sat].clone();
	//cv::addWeighted(
	//	(30 / (imginfo.image.hsv_origins[Cind::Sat])),
	//	(pos - imginfo.trackbar.vibrance),
	//	tmp,
	//	1,
	//	0,
	//	imginfo.filter.hsv_filters[Cind::Sat],
	//	CV_16S
	//);

	// 변경치 업데이트
	imginfo.trackbar.vibrance = pos;
	imginfo.changed_color_space = HSV_CHANGED;
}

void update_highlight_hue(int pos) {
	Mat tmp = imginfo.filter.hls_filters[Cind::H].clone();
	cv::addWeighted(
		(imginfo.image.hls_origins[Cind::L]),
		(double)(pos - imginfo.trackbar.highlight_hue) * 0.001,
		tmp,
		1,
		0,
		imginfo.filter.hls_filters[Cind::H],
		CV_16S
	);

	// 변경치 업데이트
	imginfo.trackbar.highlight_hue = pos;
	imginfo.changed_color_space = HLS_CHANGED;
}

void update_highlight_saturation(int pos) {
	Mat tmp = imginfo.filter.hls_filters[Cind::S].clone();
	cv::addWeighted(
		(imginfo.image.hls_origins[Cind::L]),
		double(pos - imginfo.trackbar.highlight_sat) * 0.01,
		tmp,
		1,
		0,
		imginfo.filter.hls_filters[Cind::S],
		CV_16S
	);

	// 변경치 업데이트
	imginfo.trackbar.highlight_sat = pos;
	imginfo.changed_color_space = HLS_CHANGED;
}

void update_shadow_hue(int pos) {
	Mat tmp = imginfo.filter.hls_filters[Cind::H].clone();
	Mat tmp2;
	cv::divide(15, imginfo.image.hls_origins[Cind::L], tmp2, CV_32F);

	cv::addWeighted(
		(tmp2),
		double(pos - imginfo.trackbar.highlight_hue),
		tmp,
		1,
		0,
		imginfo.filter.hls_filters[Cind::H],
		CV_16S
	);

	// 변경치 업데이트
	imginfo.trackbar.highlight_hue = pos;
	imginfo.changed_color_space = HLS_CHANGED;
}

void update_shadow_saturation(int pos) {
	Mat tmp = imginfo.filter.hls_filters[Cind::S].clone();
	cv::addWeighted(
		(100 / (imginfo.image.hls_origins[Cind::L])),
		(pos - imginfo.trackbar.highlight_sat),
		tmp,
		1,
		0,
		imginfo.filter.hls_filters[Cind::S],
		CV_16S
	);

	// 변경치 업데이트
	imginfo.trackbar.highlight_sat = pos;
	imginfo.changed_color_space = HLS_CHANGED;
}

/*********************************************************************
*	이하 동훈이 코드 
**************************************************************/

void update_tint(int pos)
{
	imginfo.filter.diff.setTo((pos - imginfo.trackbar.tint)/5.0);
	cv::add(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
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
	cv::subtract(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.clarity_mask_split[Cind::B], imginfo.filter.bgr_filters[Cind::B]);
	cv::subtract(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.clarity_mask_split[Cind::G], imginfo.filter.bgr_filters[Cind::G]);
	cv::subtract(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.clarity_mask_split[Cind::R], imginfo.filter.bgr_filters[Cind::R]);


	clarity_value = pos/(double)10.0;

	cv::addWeighted(imginfo.downsized_img,clarity_value,imginfo.filter.clarity_filter,-clarity_value,0,imginfo.filter.clarity_mask_U);
	// cout<<imginfo.filter.clarity_mask_U.type()<<endl;
	imginfo.filter.clarity_mask_U.convertTo(imginfo.filter.clarity_mask_S,CV_16SC3,0.8);
	cv::split(imginfo.filter.clarity_mask_S,imginfo.filter.clarity_mask_split);
	cv::add(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.clarity_mask_split[Cind::B], imginfo.filter.bgr_filters[Cind::B]);
	cv::add(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.clarity_mask_split[Cind::G], imginfo.filter.bgr_filters[Cind::G]);
	cv::add(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.clarity_mask_split[Cind::R], imginfo.filter.bgr_filters[Cind::R]);

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

	cv::multiply(imginfo.bgr_split[Cind::B], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::subtract(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);


	cv::multiply(imginfo.bgr_split[Cind::G], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::subtract(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);

	cv::multiply(imginfo.bgr_split[Cind::R], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::subtract(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);

	imginfo.filter.diff.setTo(b);
	cv::subtract(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);
	cv::subtract(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
	cv::subtract(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);

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

	cv::multiply(imginfo.bgr_split[Cind::B], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::add(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);


	cv::multiply(imginfo.bgr_split[Cind::G], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::add(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);

	cv::multiply(imginfo.bgr_split[Cind::R], a - 1, imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_16S);
	cv::add(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);

	imginfo.filter.diff.setTo(b);
	cv::add(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);
	cv::add(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
	cv::add(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);


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
		cv::subtract(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);
		cv::subtract(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
		cv::subtract(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
	}
	else
	{
		cv::add(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);
		cv::add(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
		cv::add(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
	}

	imginfo.filter.diff.setTo(abs(pos));
	if (pos >= 0)
	{
		cv::add(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);
		cv::add(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
		cv::add(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
	}
	else
	{
		cv::subtract(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);
		cv::subtract(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
		cv::subtract(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
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
	cv::subtract(imginfo.filter.hsv_filters[Cind::V], imginfo.filter.diff, imginfo.filter.hsv_filters[Cind::V]);

	imginfo.filter.diff = imginfo.filter.gamma_mask.clone();

	cv::pow(imginfo.filter.diff, pos, imginfo.filter.diff);
	cv::multiply(255, imginfo.filter.diff, imginfo.filter.diff);
	cv::cvtColor(imginfo.filter.diff, imginfo.filter.diff, CV_8U);
	cv::add(imginfo.filter.hsv_filters[Cind::V], imginfo.filter.diff, imginfo.filter.hsv_filters[Cind::V]);

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
	cv::add(imginfo.filter.hsv_filters[Cind::V], imginfo.filter.diff, imginfo.filter.hsv_filters[Cind::V]);

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
	cout<<imginfo.filter.diff<<endl;
	cv::multiply(imginfo.filter.diff,200,imginfo.filter.diff);
	imginfo.filter.diff.convertTo(imginfo.filter.diff,CV_32F);
	imginfo.res_split[Cind::B].convertTo(imginfo.res_split[Cind::B],CV_32F);
	imginfo.res_split[Cind::R].convertTo(imginfo.res_split[Cind::R],CV_32F);
	imginfo.res_split[Cind::G].convertTo(imginfo.res_split[Cind::G],CV_32F);

	//양이 밝게 , 음이 어둡게
	if (imginfo.trackbar.vignette > 0)
	{
		cv::add(imginfo.res_split[Cind::B], imginfo.filter.diff, imginfo.res_split[Cind::B]);
		cv::add(imginfo.res_split[Cind::G], imginfo.filter.diff, imginfo.res_split[Cind::G]);
		cv::add(imginfo.res_split[Cind::R], imginfo.filter.diff, imginfo.res_split[Cind::R]);
	}
	else if (imginfo.trackbar.vignette < 0)
	{
		cv::subtract(imginfo.res_split[Cind::B], imginfo.filter.diff, imginfo.res_split[Cind::B]);
		cv::subtract(imginfo.res_split[Cind::G], imginfo.filter.diff, imginfo.res_split[Cind::G]);
		cv::subtract(imginfo.res_split[Cind::R], imginfo.filter.diff, imginfo.res_split[Cind::R]);
	}

	cv::merge(imginfo.res_split,imginfo.res_img);
	imginfo.res_img.convertTo(imginfo.res_img,CV_8U);
	imshow("Dsd",imginfo.res_img);

	if (pos > 0)
	{
		cv::multiply(imginfo.res_split[Cind::B], imginfo.filter.diff, imginfo.res_split[Cind::B]);
		cv::multiply(imginfo.res_split[Cind::G], imginfo.filter.diff, imginfo.res_split[Cind::G]);
		cv::multiply(imginfo.res_split[Cind::R], imginfo.filter.diff, imginfo.res_split[Cind::R]);
	}
	else if (pos < 0)
	{
		cv::divide(imginfo.res_split[Cind::B], imginfo.filter.diff, imginfo.res_split[Cind::B]);
		cv::divide(imginfo.res_split[Cind::G], imginfo.filter.diff, imginfo.res_split[Cind::G]);
		cv::divide(imginfo.res_split[Cind::R], imginfo.filter.diff, imginfo.res_split[Cind::R]);
	}


	// imginfo.res_split[Cind::B].convertTo(imginfo.res_split[Cind::B],CV_8U);
	// imginfo.res_split[Cind::R].convertTo(imginfo.res_split[Cind::R],CV_8U);
	// imginfo.res_split[Cind::G].convertTo(imginfo.res_split[Cind::G],CV_8U);
	cv::convertScaleAbs(imginfo.res_split[Cind::B],imginfo.res_split[Cind::B]);
	cv::convertScaleAbs(imginfo.res_split[Cind::G],imginfo.res_split[Cind::G]);
	cv::convertScaleAbs(imginfo.res_split[Cind::R],imginfo.res_split[Cind::R]);

	
	cv::merge(imginfo.res_split,imginfo.res_img);
	
/*
	cv::split(imginfo.res_img,imginfo.res_split);
	imginfo.filter.diff = imginfo.filter.gaussian_kernel.clone();
	imginfo.res_split[Cind::B].convertTo(imginfo.res_split[Cind::B],CV_64F);
	imginfo.res_split[Cind::R].convertTo(imginfo.res_split[Cind::R],CV_64F);
	imginfo.res_split[Cind::G].convertTo(imginfo.res_split[Cind::G],CV_64F);

	//양이 밝게 , 음이 어둡게
	if (imginfo.trackbar.vignette > 0)
	{
		cv::multiply(imginfo.res_split[Cind::B], imginfo.filter.diff, imginfo.res_split[Cind::B]);
		cv::multiply(imginfo.res_split[Cind::G], imginfo.filter.diff, imginfo.res_split[Cind::G]);
		cv::multiply(imginfo.res_split[Cind::R], imginfo.filter.diff, imginfo.res_split[Cind::R]);
	}
	else if (imginfo.trackbar.vignette < 0)
	{
		cv::divide(imginfo.res_split[Cind::B], imginfo.filter.diff, imginfo.res_split[Cind::B]);
		cv::divide(imginfo.res_split[Cind::G], imginfo.filter.diff, imginfo.res_split[Cind::G]);
		cv::divide(imginfo.res_split[Cind::R], imginfo.filter.diff, imginfo.res_split[Cind::R]);
	}

	if (pos > 0)
	{
		cv::multiply(imginfo.res_split[Cind::B], imginfo.filter.diff, imginfo.res_split[Cind::B]);
		cv::multiply(imginfo.res_split[Cind::G], imginfo.filter.diff, imginfo.res_split[Cind::G]);
		cv::multiply(imginfo.res_split[Cind::R], imginfo.filter.diff, imginfo.res_split[Cind::R]);
	}
	else if (pos < 0)
	{
		cv::divide(imginfo.res_split[Cind::B], imginfo.filter.diff, imginfo.res_split[Cind::B]);
		cv::divide(imginfo.res_split[Cind::G], imginfo.filter.diff, imginfo.res_split[Cind::G]);
		cv::divide(imginfo.res_split[Cind::R], imginfo.filter.diff, imginfo.res_split[Cind::R]);
	}


	// imginfo.res_split[Cind::B].convertTo(imginfo.res_split[Cind::B],CV_8U);
	// imginfo.res_split[Cind::R].convertTo(imginfo.res_split[Cind::R],CV_8U);
	// imginfo.res_split[Cind::G].convertTo(imginfo.res_split[Cind::G],CV_8U);
	cv::convertScaleAbs(imginfo.res_split[Cind::B],imginfo.res_split[Cind::B]);
	cv::convertScaleAbs(imginfo.res_split[Cind::G],imginfo.res_split[Cind::G]);
	cv::convertScaleAbs(imginfo.res_split[Cind::R],imginfo.res_split[Cind::R]);

	
	cv::merge(imginfo.res_split,imginfo.res_img);


*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// imginfo.filter.diff = imginfo.filter.gaussian_kernel.clone();
	// cv::multiply(imginfo.filter.diff, imginfo.trackbar.vignette, imginfo.filter.diff);
	// imginfo.filter.bgr_filters[Cind::B].convertTo(imginfo.filter.bgr_filters[Cind::B],CV_64F);
	// imginfo.filter.bgr_filters[Cind::R].convertTo(imginfo.filter.bgr_filters[Cind::R],CV_64F);
	// imginfo.filter.bgr_filters[Cind::G].convertTo(imginfo.filter.bgr_filters[Cind::G],CV_64F);

	// //양이 밝게 , 음이 어둡게
	// if (imginfo.trackbar.vignette > 0)
	// {
	// 	cv::multiply(imginfo.filter.bgr_filters[Cind::B], .fimginfoilter.diff, imginfo.filter.bgr_filters[Cind::B]);
	// 	cv::multiply(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
	// 	cv::multiply(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
	// }
	// else if (imginfo.trackbar.vignette < 0)
	// {
	// 	cv::divide(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);
	// 	cv::divide(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
	// 	cv::divide(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
	// }
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[Cind::B],imginfo.filter.bgr_filters[Cind::B]);
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[Cind::G],imginfo.filter.bgr_filters[Cind::G]);
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[Cind::R],imginfo.filter.bgr_filters[Cind::R]);


	// cv::divide(imginfo.filter.diff, imginfo.trackbar.vignette, imginfo.filter.diff);
	// cv::multiply(imginfo.filter.diff, pos, imginfo.filter.diff);
	// cout<<imginfo.filter.bgr_filters[Cind::B].type()<<endl;
	// cout<<imginfo.filter.diff.type()<<endl;

	// if (pos >= 0)
	// {
	// 	cv::divide(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);
	// 	cv::divide(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
	// 	cv::divide(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
	// }
	// else if (pos < 0)
	// {
	// 	cv::multiply(imginfo.filter.bgr_filters[Cind::B], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::B]);
	// 	cv::multiply(imginfo.filter.bgr_filters[Cind::G], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::G]);
	// 	cv::multiply(imginfo.filter.bgr_filters[Cind::R], imginfo.filter.diff, imginfo.filter.bgr_filters[Cind::R]);
	// }
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[Cind::B],imginfo.filter.bgr_filters[Cind::B]);
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[Cind::G],imginfo.filter.bgr_filters[Cind::G]);
	// cv::convertScaleAbs(imginfo.filter.bgr_filters[Cind::R],imginfo.filter.bgr_filters[Cind::R]);
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