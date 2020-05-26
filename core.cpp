#include "define.h"
#include "header.h"

double GND(double x, double w, double std, double mu = 0) {
	return w * pow(EXP, -((x - mu)*(x - mu)) / (2 * std*std)) / sqrt(2 * PI*std*std);
}

double weight_per_saturation(int val, int mu) {
	return GND((double)val, 200.0, 50.0, (double)mu);
}

double weight_per_value(int val, int mu) {
	return GND((double)val, 200.0, 50.0, (double)mu);
}

void downsize_image(cv::Mat &src, cv::Mat &dst, int width, int height) {
	if (width <= src.size().width && height <= src.size().height) {
		cv::resize(src, dst, cv::Size(width, height), 0, 0, cv::INTER_AREA);
	}
	else {
		dst = src.clone();
	}
}

// return img + logo
cv::Mat get_watermarked_image(cv::Mat src_img, cv::Mat src_logo, int width, int height) {
	cv::Mat res_img, res_logo;

	cv::resize(src_logo, res_logo, src_img.size(), 0, 0, cv::INTER_AREA);
	cv::addWeighted(src_img, 1, res_logo, 0.3, 0, res_img);

	if (width && height) 
		cv::resize(res_img, res_img, cv::Size(width, height), 0, 0, cv::INTER_AREA);

	return res_img;
}

// return img + logo + filter
cv::Mat get_preview_image(
	cv::Mat& src_img, cv::Mat src_logo,
	int hue, int saturation, int lightness, int vibrance,
	int highlight_hue, int highlight_sat, int shadow_hue, int shadow_sat,
	int temperature, int tint, int brightness, int grain,
	int clarity, int exposure, int gamma, int vignette, int constrast,
	int width, int height /* for downsizing */
) {
	WorkingImgInfo preview_info;
	preview_info.init_all(src_img, width, height);

	preview_info.update_hue(hue);
	preview_info.update_saturation(saturation);
	preview_info.update_lightness(lightness);
	preview_info.update_vibrance(vibrance);
	preview_info.update_highlight_hue(highlight_hue);
	preview_info.update_highlight_saturation(highlight_sat);
	preview_info.update_shadow_hue(shadow_hue);
	preview_info.update_shadow_saturation(shadow_sat);
	preview_info.update_temperature(temperature);
	preview_info.update_tint(tint);
	//preview_info.update_brightness_and_constrast(brightness);
	preview_info.update_grain(grain);
	preview_info.update_clarity(clarity);
	preview_info.update_exposure(exposure);
	//preview_info.update_gamma(gamma); std::cout << "hiii\n";
	preview_info.update_vignette(vignette);
	//preview_info.update_brightness_and_constrast(constrast);

	preview_info.apply_filter();
	return get_watermarked_image(preview_info.image.res, src_logo, width, height);
}

/*****************************************************************************
*							applyFilter
*	add bgr filter -> convert to hsv -> add hsv filter -> convert to bgr(res)
*****************************************************************************/

void WorkingImgInfo::apply_filter() {
	this->image.hls.convertTo(this->image.hls, CV_16SC3);
	this->image.bgr.convertTo(this->image.bgr, CV_16SC3);

	// hls
	cv::merge(this->filter.hls_filters, this->filter.hls_filter);
	this->image.res.convertTo(this->image.res, CV_16SC3);
	cv::add(this->image.hls, this->filter.hls_filter, this->image.res); /**/
	this->image.res.convertTo(this->image.res, CV_8UC3);

	// bgr
	cv::cvtColor(this->image.res, this->image.res, cv::COLOR_HLS2BGR);
	cv::merge(this->filter.bgr_filters, this->filter.bgr_filter);

	this->image.res.convertTo(this->image.res, CV_16SC3);
	cv::add(this->image.res, this->filter.bgr_filter, this->image.res);
	this->image.res.convertTo(this->image.res, CV_8UC3);

	this->image.bgr.convertTo(this->image.bgr, CV_8UC3);
	this->image.hls.convertTo(this->image.hls, CV_8UC3);
}

void WorkingImgInfo::update_hue(int pos) {
	this->filter.diff.setTo(pos - this->trackbar.hue);
	cv::add(
		this->filter.hls_filters[HLSINDEX::H],
		this->filter.diff,
		this->filter.hls_filters[HLSINDEX::H]
	);
	this->trackbar.hue = pos;
}

void WorkingImgInfo::update_saturation(int pos) {
	this->filter.diff.setTo(pos - this->trackbar.saturation);
	cv::add(
		this->filter.hls_filters[HLSINDEX::S],
		this->filter.diff,
		this->filter.hls_filters[HLSINDEX::S]
	);
	this->trackbar.saturation = pos;
}

void WorkingImgInfo::update_lightness(int pos) {
	this->filter.diff.setTo(pos - this->trackbar.lightness);
	cv::add(
		this->filter.hls_filters[HLSINDEX::L],
		this->filter.diff,
		this->filter.hls_filters[HLSINDEX::L]
	);
	this->trackbar.lightness = pos;
}

void WorkingImgInfo::update_temperature(int pos) {
	this->filter.diff.setTo(abs(this->trackbar.temperature));
	if (this->trackbar.temperature >= 0)
		cv::subtract(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);
	else
		cv::subtract(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);

	this->filter.diff.setTo(abs(pos));
	if (pos >= 0)
		cv::add(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);
	else
		cv::add(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);

	this->trackbar.temperature = pos;
}

void WorkingImgInfo::update_vibrance(int pos) {
	cv::Mat w;
	cv::divide(
		this->image.hls_origins[HLSINDEX::L],
		this->image.hls_origins[HLSINDEX::S],
		w,
		(double)(pos - this->trackbar.vibrance) * 0.05,
		CV_16S
	);
	cv::add(
		this->filter.hls_filters[HLSINDEX::S],
		w,
		this->filter.hls_filters[HLSINDEX::S]
	);

	// 변경치 업데이트
	this->trackbar.vibrance = pos;
}

void WorkingImgInfo::update_highlight_hue(int pos) {
	cv::Mat tmp = this->filter.hls_filters[HLSINDEX::H].clone();
	cv::addWeighted(
		(this->image.hls_origins[HLSINDEX::L]),
		(double)(pos - this->trackbar.highlight_hue) * 0.001,
		tmp,
		1,
		0,
		this->filter.hls_filters[HLSINDEX::H],
		CV_16S
	);

	// 변경치 업데이트
	this->trackbar.highlight_hue = pos;
}

void WorkingImgInfo::update_highlight_saturation(int pos) {
	cv::Mat tmp = this->filter.hls_filters[HLSINDEX::S].clone();
	cv::addWeighted(
		(this->image.hls_origins[HLSINDEX::L]),
		double(pos - this->trackbar.highlight_sat) * 0.01,
		tmp,
		1,
		0,
		this->filter.hls_filters[HLSINDEX::S],
		CV_16S
	);

	// 변경치 업데이트
	this->trackbar.highlight_sat = pos;
}

void WorkingImgInfo::update_shadow_hue(int pos) {
	cv::Mat tmp = this->filter.hls_filters[HLSINDEX::H].clone();
	cv::Mat tmp2;
	cv::divide(15, this->image.hls_origins[HLSINDEX::L], tmp2, CV_32F);

	cv::addWeighted(
		(tmp2),
		double(pos - this->trackbar.highlight_hue),
		tmp,
		1,
		0,
		this->filter.hls_filters[HLSINDEX::H],
		CV_16S
	);

	// 변경치 업데이트
	this->trackbar.highlight_hue = pos;
}

void WorkingImgInfo::update_shadow_saturation(int pos) {
	cv::Mat tmp = this->filter.hls_filters[HLSINDEX::S].clone();
	cv::addWeighted(
		(100 / (this->image.hls_origins[HLSINDEX::L])),
		(pos - this->trackbar.highlight_sat),
		tmp,
		1,
		0,
		this->filter.hls_filters[HLSINDEX::S],
		CV_16S
	);

	// 변경치 업데이트
	this->trackbar.highlight_sat = pos;
}

/*********************************************************************
*	이하 동훈이 코드
*********************************************************************/
void WorkingImgInfo::update_tint(int pos) {
	this->filter.diff.setTo((pos - this->trackbar.tint));
	cv::add(this->filter.bgr_filters[BGRINDEX::G], this->filter.diff, this->filter.bgr_filters[BGRINDEX::G]);

	this->trackbar.tint = pos;
}

void WorkingImgInfo::update_clarity(int pos)
{
	double clarity_value;
	clarity_value = this->trackbar.clarity / (double)10.0;

	cv::addWeighted(this->image.downsized, clarity_value, this->filter.clarity_filter, -clarity_value, 0, this->filter.clarity_mask_U);
	this->filter.clarity_mask_U.convertTo(this->filter.clarity_mask_S, CV_16SC3, 0.8);
	cv::split(this->filter.clarity_mask_S, this->filter.clarity_mask_split);
	cv::subtract(this->filter.bgr_filters[BGRINDEX::B], this->filter.clarity_mask_split[BGRINDEX::B], this->filter.bgr_filters[BGRINDEX::B]);
	cv::subtract(this->filter.bgr_filters[BGRINDEX::G], this->filter.clarity_mask_split[BGRINDEX::G], this->filter.bgr_filters[BGRINDEX::G]);
	cv::subtract(this->filter.bgr_filters[BGRINDEX::R], this->filter.clarity_mask_split[BGRINDEX::R], this->filter.bgr_filters[BGRINDEX::R]);

	clarity_value = pos / (double)10.0;

	cv::addWeighted(this->image.downsized, clarity_value, this->filter.clarity_filter, -clarity_value, 0, this->filter.clarity_mask_U);
	this->filter.clarity_mask_U.convertTo(this->filter.clarity_mask_S, CV_16SC3, 0.8);
	cv::split(this->filter.clarity_mask_S, this->filter.clarity_mask_split);
	cv::add(this->filter.bgr_filters[BGRINDEX::B], this->filter.clarity_mask_split[BGRINDEX::B], this->filter.bgr_filters[BGRINDEX::B]);
	cv::add(this->filter.bgr_filters[BGRINDEX::G], this->filter.clarity_mask_split[BGRINDEX::G], this->filter.bgr_filters[BGRINDEX::G]);
	cv::add(this->filter.bgr_filters[BGRINDEX::R], this->filter.clarity_mask_split[BGRINDEX::R], this->filter.bgr_filters[BGRINDEX::R]);

	this->trackbar.clarity = pos;
}

// Refactoring : a,b 구하는건 함수로
void WorkingImgInfo::update_brightness_and_constrast(int brightness_pos, int constrast_pos)
{
	double a, b;
	if (this->trackbar.constrast > 0)
	{
		double delta = MAX_7B_F * this->trackbar.constrast / MAX_8B_F;
		a = MAX_8B_F / (MAX_8B_F - delta * 2);
		b = a * (this->trackbar.brightness - delta);
	}
	else
	{
		double delta = -MAX_7B_F * this->trackbar.constrast / MAX_8B_F;
		a = (MAX_8B_F - delta * 2) / MAX_8B_F;
		b = a * this->trackbar.brightness + delta;
	}

	cv::multiply(this->image.bgr_origins[BGRINDEX::B], a - 1, this->filter.diff);
	this->filter.diff.convertTo(this->filter.diff, CV_16S);
	cv::subtract(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);


	cv::multiply(this->image.bgr_origins[BGRINDEX::G], a - 1, this->filter.diff);
	this->filter.diff.convertTo(this->filter.diff, CV_16S);
	cv::subtract(this->filter.bgr_filters[BGRINDEX::G], this->filter.diff, this->filter.bgr_filters[BGRINDEX::G]);

	cv::multiply(this->image.bgr_origins[BGRINDEX::R], a - 1, this->filter.diff);
	this->filter.diff.convertTo(this->filter.diff, CV_16S);
	cv::subtract(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);

	this->filter.diff.setTo(b);
	cv::subtract(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);
	cv::subtract(this->filter.bgr_filters[BGRINDEX::G], this->filter.diff, this->filter.bgr_filters[BGRINDEX::G]);
	cv::subtract(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);

	if (constrast_pos > 0)
	{
		double delta = MAX_7B_F * constrast_pos / MAX_8B_F;
		a = MAX_8B_F / (MAX_8B_F - delta * 2);
		b = a * (brightness_pos - delta);
	}
	else
	{
		double delta = -MAX_7B_F * constrast_pos / MAX_8B_F;
		a = (MAX_8B_F - delta * 2) / MAX_8B_F;
		b = a * brightness_pos + delta;
	}

	cv::multiply(this->image.bgr_origins[BGRINDEX::B], a - 1, this->filter.diff);
	this->filter.diff.convertTo(this->filter.diff, CV_16S);
	cv::add(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);


	cv::multiply(this->image.bgr_origins[BGRINDEX::G], a - 1, this->filter.diff);
	this->filter.diff.convertTo(this->filter.diff, CV_16S);
	cv::add(this->filter.bgr_filters[BGRINDEX::G], this->filter.diff, this->filter.bgr_filters[BGRINDEX::G]);

	cv::multiply(this->image.bgr_origins[BGRINDEX::R], a - 1, this->filter.diff);
	this->filter.diff.convertTo(this->filter.diff, CV_16S);
	cv::add(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);

	this->filter.diff.setTo(b);
	cv::add(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);
	cv::add(this->filter.bgr_filters[BGRINDEX::G], this->filter.diff, this->filter.bgr_filters[BGRINDEX::G]);
	cv::add(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);

	this->trackbar.brightness = brightness_pos;
	this->trackbar.constrast = constrast_pos;
}

void WorkingImgInfo::update_exposure(int pos)
{
	this->filter.diff.setTo(abs(this->trackbar.exposure));

	if (this->trackbar.exposure >= 0)
	{
		cv::subtract(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);
		cv::subtract(this->filter.bgr_filters[BGRINDEX::G], this->filter.diff, this->filter.bgr_filters[BGRINDEX::G]);
		cv::subtract(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);
	}
	else
	{
		cv::add(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);
		cv::add(this->filter.bgr_filters[BGRINDEX::G], this->filter.diff, this->filter.bgr_filters[BGRINDEX::G]);
		cv::add(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);
	}

	this->filter.diff.setTo(abs(pos));
	if (pos >= 0)
	{
		cv::add(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);
		cv::add(this->filter.bgr_filters[BGRINDEX::G], this->filter.diff, this->filter.bgr_filters[BGRINDEX::G]);
		cv::add(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);
	}
	else
	{
		cv::subtract(this->filter.bgr_filters[BGRINDEX::B], this->filter.diff, this->filter.bgr_filters[BGRINDEX::B]);
		cv::subtract(this->filter.bgr_filters[BGRINDEX::G], this->filter.diff, this->filter.bgr_filters[BGRINDEX::G]);
		cv::subtract(this->filter.bgr_filters[BGRINDEX::R], this->filter.diff, this->filter.bgr_filters[BGRINDEX::R]);
	}
	this->trackbar.exposure = pos;
}

// float 처리를 해야 가능
void WorkingImgInfo::update_gamma(int pos)
{
	// double gammaValue=gamma/100.0;
	// double inv_gamma=1/gammaValue;

	cv::pow(this->filter.diff, -this->trackbar.gamma, this->filter.diff);
	cv::multiply(255, this->filter.diff, this->filter.diff);
	cv::cvtColor(this->filter.diff, this->filter.diff, CV_8U);
	cv::subtract(this->filter.hsv_filters[HSVINDEX::V], this->filter.diff, this->filter.hsv_filters[HSVINDEX::V]);

	this->filter.diff = this->filter.gamma_mask.clone();

	cv::pow(this->filter.diff, pos, this->filter.diff);
	cv::multiply(255, this->filter.diff, this->filter.diff);
	cv::cvtColor(this->filter.diff, this->filter.diff, CV_8U);
	cv::add(this->filter.hsv_filters[HSVINDEX::V], this->filter.diff, this->filter.hsv_filters[HSVINDEX::V]);

	cv::cvtColor(this->filter.diff, this->filter.diff, CV_16S);

	this->trackbar.gamma = pos;
}

void WorkingImgInfo::update_grain(int pos)
{
	this->filter.diff.convertTo(this->filter.diff, CV_32F);
	this->filter.diff = this->filter.grain_mask.clone();

	cv::multiply(this->filter.diff, (pos - this->trackbar.grain) / 5.0, this->filter.diff);
	this->filter.diff.convertTo(this->filter.diff, CV_16S);
	cv::add(this->filter.hsv_filters[HSVINDEX::V], this->filter.diff, this->filter.hsv_filters[HSVINDEX::V]);

	this->trackbar.grain = pos;
}

void WorkingImgInfo::update_vignette(int pos)
{
	this->filter.diff = this->filter.gaussian_kernel.clone();

	// 양이 밝게 , 음이 어둡게
	cv::multiply(this->filter.diff, abs(this->trackbar.vignette)*0.01, this->filter.diff);
	if (this->trackbar.vignette > 0)
	{
		cv::subtract(this->filter.hsv_filters[HSVINDEX::V], this->filter.diff, this->filter.hsv_filters[HSVINDEX::V]);
	}
	else if (this->trackbar.vignette < 0)
	{
		cv::add(this->filter.hsv_filters[HSVINDEX::V], this->filter.diff, this->filter.hsv_filters[HSVINDEX::V]);
	}

	this->filter.diff = this->filter.gaussian_kernel.clone();
	cv::multiply(this->filter.diff, abs(pos)*0.01, this->filter.diff);
	if (pos > 0)
	{
		cv::add(this->filter.hsv_filters[HSVINDEX::V], this->filter.diff, this->filter.hsv_filters[HSVINDEX::V]);
	}
	else if (pos < 0)
	{
		cv::subtract(this->filter.hsv_filters[HSVINDEX::V], this->filter.diff, this->filter.hsv_filters[HSVINDEX::V]);
	}

	this->trackbar.vignette = pos;
}