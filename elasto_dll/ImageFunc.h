//////////////////////////////////////////////////////////////////////////
//

#pragma  once

#include "opencv\cv.h"

void MakeImage(const CvMat   *psrc, const char *filename);
void MakeImage(const cv::Mat *pImage, const char *filename);
void MakeImage(const cv::Mat &image, const char *filename);
