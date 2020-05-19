#pragma once

/*********************************************************************
*	??
*********************************************************************/
#define ORANGE 15
#define YELLOW 30
#define CHARTREUSE 45
#define GREEN 60
#define SPRING 75
#define CYAN 90
#define AZURE 105
#define BLUE 120
#define VIOLET 135
#define MAGENTA 150
#define ROSE 175
#define RED 0

#define RED_IND 0
#define ORANGE_IND 1
#define YELLOW_IND 2
#define GREEN_IND 3
#define BLUE_IND 4
#define VIOLET_IND 5

#define COLOR_NUM 6

#define HUE_MIN 0
#define HUE_MAX 179
#define SAT_MIN 0
#define SAT_MAX 255
#define VAL_MIN 0
#define VAL_MAX 255

#define HUE_RED_LB 171
#define HUE_RED_UB 7
#define HUE_ORANGE_LB 8
#define HUE_ORANGE_UB 20
#define HUE_YELLOW_LB 21
#define HUE_YELLOW_UB 35
#define HUE_GREEN_LB 36
#define HUE_GREEN_UB 80
#define HUE_BLUE_LB 81
#define HUE_BLUE_UB 127
#define HUE_VIOLET_LB 128
#define HUE_VIOLET_UB 170

#define RANGE_RED(x) ((HUE_RED_LB <= x && x <= HUE_MAX) || (HUE_MIN <= x && x <= HUE_RED_UB))
#define RANGE_ORANGE(x) (HUE_ORANGE_LB <= x && x <= HUE_ORANGE_UB)
#define RANGE_YELLOW(x) (HUE_YELLOW_LB <= x && x <= HUE_YELLOW_UB)
#define RANGE_GREEN(x) (HUE_GREEN_LB <= x && x <= HUE_GREEN_UB)
#define RANGE_BLUE(x) (HUE_BLUE_LB <= x && x <= HUE_BLUE_UB)
#define RANGE_VIOLET(x) (HUE_VIOLET_LB <= x && x <= HUE_VIOLET_UB)

#define TEST_WINDOW "Test"
#define SET_WINDOW "Set"

#define TRACKBAR_MIN 0
#define TRACKBAR_MID 100
#define TRACKBAR_MAX 200

#define PI 3.141592
#define EXP exp(1)

#define CHECK_TIME_START(start) start = cv::getTickCount()
#define CHECK_TIME_END(end) end = cv::getTickCount()
#define CHECK_TIME_PRINT(str, start, end) printf("%s%f(s)\n", str, static_cast<double>(end - start) / cv::getTickFrequency())

/*********************************************************************
*	??
*********************************************************************/
#define ZERO 0
#define ONE 1
#define FIVE 5
#define TEN 10
#define HUNDRED 100
#define TWO_HUNDRED 200
#define MAX_8B_F 255.0
#define MAX_8B_I 255
#define MAX_7B_F 128.0

#define DISTANCE 10
#define SIGMA_COLOR 50
#define SIGMA_SPACE 50

#define ESC 27

#define GRAY_MIN 0
#define GRAY_MAX 1
#define GRAY_MID 0

#define BRIGHTNESS_MIN 0
#define BRIGHTNESS_MAX 200
#define BRIGHTNESS_MID 100

#define CONSTRAST_MIN 0
#define CONSTRAST_MAX 200
#define CONSTRAST_MID 100

#define TINT_MIN 0
#define TINT_MAX 200
#define TINT_MID 100

#define CLARITY_MIN 0
#define CLARITY_MAX 200
#define CLARITY_MID 100

#define GAMMA_MIN 0
#define GAMMA_MAX 200
#define GAMMA_MID 0

#define EXPOSURE_MIN 0
#define EXPOSURE_MAX 200
#define EXPOSURE_MID 100

#define VIGNETTE_MIN 0
#define VIGNETTE_MAX 200
#define VIGNETTE_MID 100

#define GRAIN_MIN 0
#define GRAIN_MAX 50
#define GRAIN_MID 0

#define TINTVECTOR Scalar(0, 1, 0)

#define WINDOW_NAME "Doroddong Test"

#define KERNEL_SIZE 7

#define BGR_CHANGED 1
#define HLS_CHANGED 2
#define HSV_CHANGED 3