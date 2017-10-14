//////////////////////////////////////////////////////////////////////////
//

#include "stdafx.h"
#include "ImageFunc.h"
#include "opencv/highgui.h"

//////////////////////////////////////////////////////////////////////////
// �ڲ���psrc������ת�ò���
//////////////////////////////////////////////////////////////////////////
void MakeImage(const CvMat *psrc, const char *filename)
{
	if (!filename)  return;

	CvMat *pmat = cvCreateMat(psrc->cols, psrc->rows, psrc->type);

	cvTranspose(psrc, pmat);
	IplImage *pimage = cvCreateImage(cvGetSize(pmat), IPL_DEPTH_32F, 3);
	cvCvtColor(pmat, pimage, CV_GRAY2BGR);

	cvSaveImage(filename, pimage);

	cvReleaseImage(&pimage);
	cvReleaseMat(&pmat);
}

void MakeImage(const cv::Mat *pImage, const char *filename)
{
	CvMat  tmp = *pImage;
	MakeImage(&tmp, filename);
}

void MakeImage(const cv::Mat &image, const char *filename)
{
	CvMat  tmp = image;
	MakeImage(&tmp, filename);
}

