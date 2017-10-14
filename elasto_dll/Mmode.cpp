#include "stdafx.h"
#include "filter.h"
#include "Mmode.h"
#include "opencv/cxcore.h"

namespace mmode
{
static mmode::Filter s_flt;

SMModeInput s_MModeParam;

static CvMat *s_MModeImage = 0;
static CvMat *s_matGrayImage = 0;

void Initialize(const SMModeInput &param)
{
	s_flt.setFilterParam("mbandpass.txt");
	s_MModeParam = param;
	s_MModeImage = cvCreateMat(param.rows, param.cols, CV_32FC1);
	cvZero(s_MModeImage);

	s_matGrayImage = cvCreateMat(param.rows, 512, CV_32FC1);
	cvZero(s_matGrayImage);
}

void Release()
{
	if (s_MModeImage)
	{
		cvReleaseMat(&s_MModeImage);
	}
	if (s_matGrayImage)
	{
		cvReleaseMat(&s_matGrayImage);
	}
}

void DoEnvelop(const float *rf, int n, const char *file_hilber, const char *file_gray)
{
	std::string filename1;
	std::string filename2;
	filename1 = file_hilber;
	filename2 = file_gray;
	const float *inData = rf;
	float *outData    = new float[n];
	float *outHilbert = new float[n];
	s_flt.doFilter(inData, outData, n);
	s_flt.hilbert(filename1, outData, outHilbert, n);

	// 从s_MModeParam第二行开始，每行数据迁移到前一行；第一行的数据被抛弃；
	int i;
	for (i = 1; i < s_MModeImage->rows; i++)
	{
		memcpy(s_MModeImage->data.ptr + (i - 1) * s_MModeImage->step, 
			   s_MModeImage->data.ptr + i * s_MModeImage->step, sizeof(float) * n);
	}
	memcpy(s_MModeImage->data.ptr + (s_MModeParam.rows - 1) * s_MModeImage->step, outHilbert, sizeof(float) * n);//新的数据线放在最后一行的位置

#if 0

	CvMat *pmatTran = cvCreateMat(s_MModeImage->cols, s_MModeImage->rows, CV_32FC1);
	cvTranspose(s_MModeImage, pmatTran);
#else
	CvMat *pmatTran = cvCreateMat(s_matGrayImage->cols, s_matGrayImage->rows, CV_32FC1);
	float *pf32 = new float[s_matGrayImage->cols];
    int    step = (int)floor((double)s_MModeParam.cols / s_matGrayImage->cols);
	for (i = 0; i < s_matGrayImage->cols; i++)
	{
		*(pf32 + i) = *(outHilbert  + i * step);
	}

	for (i = 1; i <  s_matGrayImage->rows; i++)
	{
		memcpy(s_matGrayImage->data.ptr + (i - 1) * s_matGrayImage->step, 
			  s_matGrayImage->data.ptr + i * s_matGrayImage->step, sizeof(float) * s_matGrayImage->cols);
	}
	memcpy(s_matGrayImage->data.ptr + (s_matGrayImage->rows - 1) * s_matGrayImage->step, pf32, 
		   sizeof(float) * s_matGrayImage->cols);//新的数据线放在最后一行的位置
	delete [] pf32;
	cvTranspose(s_matGrayImage, pmatTran);

#endif
	s_flt.grayImage(filename2, pmatTran, s_MModeParam.nDyn);
	cvReleaseMat(&pmatTran);

	delete [] outHilbert;
	delete [] outData;
}

void DoEnvelop2(const CvMat *pmatRF, const char *file_hilber, const char *file_gray)
{
	std::string filename1;
	std::string filename2;
	filename1 = file_hilber;
	filename2 = file_gray;

	CvMat *pmatOutput = cvCreateMat(pmatRF->rows, pmatRF->cols, CV_32FC1);
	cvZero(pmatOutput);

	float *inData     = 0;
	float *outData    = new float[pmatRF->cols];
	float *outHilbert = new float[pmatRF->cols];
	for(int i = 0; i < pmatRF->rows; i++)
	{
		inData = (float*)(pmatRF->data.ptr + pmatRF->step * i);

		s_flt.doFilter(inData, outData, pmatRF->cols);
	    s_flt.hilbert(filename1, outData, outHilbert, pmatRF->cols);
		memcpy(pmatOutput->data.ptr + i * pmatOutput->step, outHilbert, sizeof(float) * pmatOutput->cols);
	}

	CvMat *pmatTran = cvCreateMat(pmatRF->cols, pmatRF->rows, CV_32FC1);
	cvTranspose(pmatOutput, pmatTran);
	s_flt.grayImage(filename2, pmatTran, s_MModeParam.nDyn);
	cvReleaseMat(&pmatOutput);
	cvReleaseMat(&pmatTran);

	delete [] outData;
	delete [] outHilbert;
}

}
