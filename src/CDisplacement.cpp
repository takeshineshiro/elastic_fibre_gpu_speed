#include "stdafx.h"

#include "CDisplacement.h"
#include "Elasto.h"
#include "CElasto.h"
#include <stdio.h>
#include <iostream>
#include "opencv/cv.h"
#include "opencv2/gpu/gpu.hpp"
#include <opencv2/gpu/device/common.hpp>

#include <climits>

#include "ElastoCudaExt.h"


using namespace cv;
using namespace cv::gpu;
using namespace cv::gpu::device;

#define SIGN(x) (((x)>=0)?(1):(0))

const float PI = 3.141592627f;

extern ConfigParam g_paramConfig;

void  min_max_loc(const CvMat &img, float *min_val, float *max_val, CvPoint *min_loc = 0, CvPoint *max_loc = 0);

template<typename T> PtrStepSz<T> make_PtrStepSz(void *data, int rows, int cols, int step)
{
    return PtrStepSz<T>(rows, cols, (T*)data, step);
}

//////////////////////////////////////////////////////////////////////////
//  ���չ�ʽ���Լ�����ʵ�֣�ÿ��ѭ����������ģ��templtÿ�����ص�ƽ��ֵ���Լ��������image�ڲ�ĳ��w��h���������ƽ��ֵ
//  ģ��ļ���ÿ�ζ��ظ��ˡ�ͬʱ
//  ����ÿ��ֻ�ƶ�step, ��image�����󣩾����������Ԫ����ÿ��ѭ���ظ����㣻
//  ���ܶ���ȷ�����Ǽ����ʱ��cvMatchTemplate��ܶ�
void  match_template_tm_ccorr_normed_32f(const cv::Mat &image, const cv::Mat &templt, cv::Mat &result)
{
    int i, j, w, h;
    h = image.rows - templt.rows + 1;
    w = image.cols - templt.cols + 1;
    result.create(h, w, image.type());

    float  sum       = 0.0f;
    float  img_sum   = 0.0f;
    float  tmpl_sum  = 0.0f;
    PtrStepSzf  ptrImg = make_PtrStepSz<float>(image.data, image.rows, image.cols, image.step[0]);
    PtrStepSzf  ptrTemplt(templt.rows, templt.cols, (float*)templt.data, templt.step[0]); 
    PtrStepSzf  ptrResult(result.rows, result.cols, (float*)result.data, result.step[0]);

    for (i = 0; i < image.rows - templt.rows + 1; i++)
    {
        for (j = 0; j < image.cols - templt.cols + 1; j++)
        {
            sum      = 0.0f;
            img_sum  = 0.0f;
            tmpl_sum = 0.0f;
            for (h = 0; h < templt.rows; h++)
            {
                for (w = 0; w < templt.cols; w++)
                {
                    sum += ptrImg(i + h, j + w) * ptrTemplt(h, w);
                    img_sum  += ptrImg(i + h, j + w) * ptrImg(i + h, j + w);
                    tmpl_sum += ptrTemplt(h, w) * ptrTemplt(h, w);
                }
            }
            result.ptr<float>(i)[j] = sum / sqrtf(img_sum * tmpl_sum);
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// ��min_max_loc_32f�����ϼ򻯣�ֱ�Ӽ��㡣
// ��������֤�� ������ȷ��
void  min_max_loc_32f(const cv::Mat &img, float *min_val, float *max_val, CvPoint *min_loc, CvPoint *max_loc)
{
    float  fminval = std::numeric_limits<float>::infinity();
    float  fmaxval = -fminval;

    int  rows = img.rows;
    int  cols = img.cols;
    float val;
    CvPoint min_idx, max_idx;

    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            val = img.ptr<float>(i)[j];
            if (val > fmaxval)
            {
                fmaxval = val;
                max_idx.x = j;
                max_idx.y = i;
            }
            if (val < fminval)
            {
                fminval = val;
                min_idx.x = j;
                min_idx.y = i;
            }
        }
    }

    if (min_val)
    {
        *min_val = fminval;
    }
    if (max_val)
    {
        *max_val = fmaxval;
    }
    if (min_loc)
    {
        *min_loc = min_idx;
    }
    if (max_loc)
    {
        *max_loc = max_idx;
    }
}

CDisplacement::CDisplacement()
{
}

CDisplacement::~CDisplacement()
{
}

void CDisplacement::Do()
{
	if (ElastoWorker::Instance().CudaIsCapable())
	{
		DisplacementAlgorithm_cuda();
	}
    else
    {
#if USE_CPP
        DisplacementAlgorithm_cpp();
#else
        DisplacementAlgorithm_c();
#endif
    }
}

void CDisplacement::SingularFilter(cv::Mat &dispMat, int threshold)
{
	//��ɾ�� 2009-10-29 ³���Բ���
	for (int i = 1; i < dispMat.rows - 1; ++i)   //λ��ƽ����ȥ����㣬�Ի����ϵ��Ϊ���ݡ�
	{
		for (int j = 0; j < dispMat.cols - 1; ++j)
		{
			if (*dispMat.ptr<float>(i, j) > threshold)
			{
				*dispMat.ptr<float>(i, j) = *dispMat.ptr<float>(i - 1, j);  
			}
		}
	}
}

void CDisplacement::Rel2AbsDisp(cv::Mat &dispMat)
{
	for (int i = 0; i < dispMat.cols; ++i)
	{
		for(int j = 0; j < dispMat.rows - 1; ++j)
		{
			*(dispMat.ptr<float>(j + 1, i)) = (*dispMat.ptr<float>(j + 1, i)) + (*dispMat.ptr<float>(j, i));
		}
	}
}

void CDisplacement::Rel2AbsDisp_cuda(cv::Mat &dispMat)
{
	elasto_cuda::Wrapper::Instance().Rel2AbsDisp(dispMat);
}


void CDisplacement::MeanFilter(cv::Mat &dispMat, int steps)
{
	int N = steps;
	float sum = 0.0f;
	cv::Mat tempMat = cv::Mat(dispMat.rows, dispMat.cols + N - 1, dispMat.type());
	for (int j = 0; j < N - 1; j++)
	{
		for (int i = 0; i < dispMat.rows; i++)
		{
			*tempMat.ptr<float>(i, j) = *dispMat.ptr<float>(i, 0);
		}
	}

	for (int j = N - 1; j < dispMat.cols + N - 1; j++)
	{
		for (int i = 0; i < dispMat.rows; i++)
		{
			*tempMat.ptr<float>(i, j) = *dispMat.ptr<float>(i, j - N + 1);
		}
	}

	for (int i = 0; i < dispMat.rows; i++)
	{
		for (int j = 0; j < dispMat.cols; j++)
		{
			for (int k = j; k < j + N; k++)
			{
				sum += *tempMat.ptr<float>(i ,k);
			}
			*dispMat.ptr<float>(i, j) = sum / N;
			sum = 0.0f;
		}
	}
}

void CDisplacement::MeanFilter_cuda(cv::Mat &disp, int steps)
{
	elasto_cuda::Wrapper::Instance().MeanFilter(disp, steps);
}

void CDisplacement::MatchFilter(cv::Mat &dispMat, cv::Mat &outDataMat)
{
	//����ӳ���match filter,��ǿλ���ź�
	double match[78] = {0,0,0,0,0,0,0.0941,0.1874,0.279,0.3682,0.454,0.5359,0.613,0.6846,0.7502,0.8091,0.8608,0.9049,0.9409,0.9686,0.9877,0.998,0.9995,
		0.9921,0.9759,0.951,0.9177,0.8762,0.827,0.7704,0.7069,0.6372,0.5619,0.4815,0.3969,0.3087,0.2179,0.125,0.0311,-0.0631,-0.1568,
		-0.249,-0.3391,-0.4261,-0.5094,-0.5881,-0.6616,-0.7292,-0.7904,-0.8446,-0.8912,-0.9299,-0.9604,-0.9824,-0.9956,-1,-0.9955,
		-0.9822,-0.9602,-0.9296,-0.8908,-0.844,-0.7898,-0.7286,-0.6609,-0.5873,-0.5085,-0.4253,-0.3382,-0.2481,-0.1558,-0.0622,0,0,0,0,0,0};

	outDataMat.create(dispMat.rows, dispMat.cols, dispMat.type());
	double tmp = 0;
	for (int i = 0; i < dispMat.cols; ++i)
	{
		for (int j = 0; j < dispMat.rows; ++j)
		{
			for (int k = 0; k <= j; ++k)
			{
				tmp += (*dispMat.ptr<float>(k, i)) * match[(j - k) < 78 ? (j - k) : 0];  /*���*/
			}
			*outDataMat.ptr<float>(j, i) = (float)tmp;
			tmp = 0;
		}
	}
}

void CDisplacement::MatchFilter_cuda(cv::Mat &dispMat, cv::Mat &outDataMat)
{
	elasto_cuda::Wrapper::Instance().MatchFilter(dispMat);
	outDataMat = dispMat.clone();
}

void CDisplacement::DisplacementAlgorithm_cpp()
{
    cv::Mat &outDataMat      = ElastoWorker::Instance().GetOutmat();
    ConfigParam &configParam = ElastoWorker::Instance().GetConfigParam();
    int    multiWin = 2;   //�󴰿ڶ�С���ڵı���

    int    winSize  = configParam.windowHW;
    int    stepSize = configParam.step;

    int     winNum      = (outDataMat.cols - multiWin * winSize) / stepSize;  //������Ҫƥ��Ķ���
    cv::Mat dispMat(outDataMat.rows - 1, winNum, CV_32FC1);     //���ڱ���λ�Ƶľ���ƥ��ó�������λ�ƾ�����ֵ��Ӧ��Ϊ������
    cv::Mat templateMat; //= cvCreateMatHeader(1, winSize, CV_32FC1);  //����ƥ���ģ�� 
    cv::Mat objectMat;   //= cvCreateMatHeader(1, multiWin * winSize, CV_32FC1);  //����ƥ���Ŀ��
    cv::Mat resultMat;  //���ڴ��ƥ����

    double    min, max;     //ƥ�����е����ֵ
    cv::Point max_loc;      //ƥ�������ֵ����λ��
    double    pmax, nmax;   //���ֵ��ǰ������ֵ
    double    displacement; //�洢λ��ֵ

    //#pragma  omp  parallel for schedule(dynamic)
    float *p;

    for (int i = 0; i < dispMat.rows; ++i)   //��λ��
    {
        for (int j = 0; j < dispMat.cols; ++j)
        {   
            templateMat = outDataMat(cv::Rect((multiWin - 1) * winSize / 2 + j * stepSize, i, winSize, 1));
            objectMat = outDataMat(cv::Rect(j * stepSize, i + 1, multiWin * winSize, 1));

            cv::matchTemplate(objectMat, templateMat, resultMat, CV_TM_CCORR_NORMED); //ƥ��

            cv::minMaxLoc(resultMat, &min, &max, NULL, &max_loc);
			/*
            {
			    // �����Լ�д���㷨���м��㣬��Ӧ�ú�opencv�Ľ����ͬ
                match_template_tm_ccorr_normed_32f(objectMat, templateMat, resultMat);
                float min_val, max_val;
                CvPoint maxLoc;
                min_max_loc_32f(resultMat, &min_val, &max_val, NULL, &maxLoc);
                min = min_val;
                max = max_val;
                max_loc.x = maxLoc.x;
                max_loc.y = maxLoc.y;
            }
			*/

            pmax = *resultMat.ptr<float>(0, (max_loc.x - 1) < 0 ? 0 : (max_loc.x - 1));
            nmax = *resultMat.ptr<float>(0, (max_loc.x + 1) < resultMat.cols ? (max_loc.x + 1) : max_loc.x);

            displacement = (multiWin - 1) * winSize / 2 - max_loc.x - (pmax - nmax) / (2 * (pmax - 2 * max + nmax));   //��ֵ�õ�λ��
            p = dispMat.ptr<float>(i, j);
            *p = static_cast<float>(displacement); //�����ŵ�λ�ƾ�����
        }
    }

    //λ��ƽ����ȥ����㣬�Ի����ϵ��Ϊ���ݡ�
	SingularFilter(dispMat, 12);

    //λ�Ƶ���
	Rel2AbsDisp(dispMat);
	//(ElastoWorker::Instance().CudaIsCapable()) ? Rel2AbsDisp_cuda(dispMat) : Rel2AbsDisp(dispMat);

	//��ֵ�˲�
	//(ElastoWorker::Instance().CudaIsCapable()) ? MeanFilter_cuda(dispMat, 100) : MeanFilter(dispMat, 100);
	MeanFilter(dispMat, 100);

    //match filter,��ǿλ���ź�
	MatchFilter(dispMat, outDataMat);
	//(ElastoWorker::Instance().CudaIsCapable()) ? MatchFilter_cuda(dispMat, outDataMat) : MatchFilter(dispMat, outDataMat);
}

void CDisplacement::DisplacementAlgorithm_cuda()
{
    cv::Mat &outDataMat      = ElastoWorker::Instance().GetOutmat();
    ConfigParam &configParam = ElastoWorker::Instance().GetConfigParam();
    int    multiWin = 2;   //�󴰿ڶ�С���ڵı���

    int    winSize  = configParam.windowHW;
    int    stepSize = configParam.step;

    int     winNum      = (outDataMat.cols - multiWin * winSize) / stepSize;  //������Ҫƥ��Ķ���
	cv::Mat dispMat;
	elasto_cuda::Wrapper::Instance().CalcDisplacement(outDataMat, dispMat, winSize, stepSize, multiWin);

    //³���Բ���
    //λ��ƽ����ȥ����㣬�Ի����ϵ��Ϊ���ݡ�
	SingularFilter(outDataMat, 12);

    //λ�Ƶ���
	Rel2AbsDisp_cuda(dispMat);

    //��ֵ�˲�
    MeanFilter_cuda(dispMat, 100);

    //match filter,��ǿλ���ź�
	MatchFilter_cuda(dispMat, outDataMat);
}

//////////////////////////////////////////////////////////////////////////
// opencv��C�ӿں�������಻����ĵط�������������ڴ�й©��
// ����Ĵ���templateMat, objectMat, �Ҹĳ���cvCreateMatHeader, Ӧ�ø�����
// ԭ���Ĵ����ƺ��п����ڴ�й¶�� data.ptrָ���ֵ�Ѿ���д�ˡ���ֵ���releaseʱȴû�б���
// ������opencv�ڲ������ڷ�������Ͽ��ǲ��ܣ�������û���㹻���opencv�����
//     ���
//////////////////////////////////////////////////////////////////////////
void CDisplacement::DisplacementAlgorithm_c()
{
    cv::Mat &outMat          = ElastoWorker::Instance().GetOutmat();
    CvMat  outDataMat        = outMat;
    ConfigParam &configParam = ElastoWorker::Instance().GetConfigParam();

    int    winSize  = configParam.windowHW;
    int    stepSize = configParam.step;

    int    multiWin = 2;   //�󴰿ڶ�С���ڵı���
    int    winNum   = static_cast<int>(outDataMat.cols - multiWin * winSize) / stepSize;  //������Ҫƥ��Ķ���
    CvMat *disMat      = cvCreateMat(outDataMat.rows - 1, winNum, CV_32FC1);     //���ڱ���λ�Ƶľ���ƥ��ó�������λ�ƾ�����ֵ��Ӧ��Ϊ������
    CvMat *templateMat = cvCreateMatHeader(1, winSize, CV_32FC1);  //����ƥ���ģ�� 
    CvMat *objectMat   = cvCreateMatHeader(1, multiWin * winSize, CV_32FC1);  //����ƥ���Ŀ��
    CvMat *resultMat   = cvCreateMat(1, (multiWin - 1) * winSize + 1, CV_32FC1);  //���ڴ��ƥ����

    //CvMat *maxMat      = cvCreateMat(outDataMat->rows - 1, winNum, CV_32FC1);    //���ڴ洢ÿ��ƥ������ƥ����

    double min, max;  //ƥ�����е����ֵ
    CvPoint max_loc;  //ƥ�������ֵ����λ��
    double pmax, nmax; //���ֵ��ǰ������ֵ
    double displacement; //�洢λ��ֵ
    //#pragma  omp  parallel for schedule(dynamic)

    for (int i = 0; i < disMat->rows; ++i)   //��λ��
    {
        for (int j = 0; j < disMat->cols; ++j)
        {         /* ->data.ptr��ʾ���ݵ���ʼλ�� */
            templateMat->data.ptr = outDataMat.data.ptr + i * outDataMat.step 
                + static_cast<int>(sizeof(float) * ((multiWin - 1) * winSize / 2 + j * stepSize)); 
            objectMat->data.ptr   = outDataMat.data.ptr + (i + 1) * outDataMat.step + static_cast<int>(sizeof(float) * (j * stepSize)/* + 0.5*/);
            cvMatchTemplate(objectMat, templateMat, resultMat, CV_TM_CCORR_NORMED); //ƥ��
			//match_template_tm_ccorr_normed_32f(cv::Mat(objectMat), cv::Mat(templateMat), cv::Mat(resultMat));
            cvMinMaxLoc(resultMat, &min, &max, NULL, &max_loc);
            //pmax = *static_cast<float*>(static_cast<void*>(resultMat->data.ptr + sizeof(float)*max_loc.x - sizeof(float)));  //ȡ���ֵ��ǰһ��ֵ
            //nmax = *static_cast<float*>(static_cast<void*>(resultMat->data.ptr + sizeof(float)*max_loc.x + sizeof(float)));  //ȡ���ֵ�ĺ�һ��ֵ
			pmax = *static_cast<float*>(static_cast<void*>(resultMat->data.ptr 
				                                           + sizeof(float) * ((max_loc.x - 1) < 0 ? 0 : max_loc.x - 1)));
			nmax = *static_cast<float*>(static_cast<void*>(resultMat->data.ptr 
				                                           + sizeof(float) * ((max_loc.x + 1) < resultMat->cols ? max_loc.x + 1 : max_loc.x)));  //ȡ���ֵ�ĺ�һ��ֵ

            displacement = (multiWin - 1) * winSize / 2 - max_loc.x - (pmax - nmax) / (2 * (pmax - 2 * max + nmax));   //��ֵ�õ�λ��

            CV_MAT_ELEM(*disMat,float, i, j) = static_cast<float>(displacement); //�����ŵ�λ�ƾ�����
        }
    }

    //��ɾ�� 2009-10-29 ³���Բ���
    for (int i = 1; i < disMat->rows - 1; ++i)   //λ��ƽ����ȥ����㣬�Ի����ϵ��Ϊ���ݡ�
    {
        for (int j = 0; j < disMat->cols - 1; ++j)
        {
            if (abs(CV_MAT_ELEM(*disMat, float, i, j)) > 12)
            {
                CV_MAT_ELEM(*disMat, float, i, j) = CV_MAT_ELEM(*disMat, float, i - 1, j);  
            }
        }
    }

    //λ�Ƶ���
    for (int i = 0; i < disMat->cols; ++i)
    {
        for(int j = 0; j < disMat->rows - 1; ++j)
        {
            CV_MAT_ELEM(*disMat, float, j + 1, i) = CV_MAT_ELEM(*disMat, float, j + 1, i) + CV_MAT_ELEM(*disMat, float, j, i);
        }
    }

    //��ֵ�˲�
    int N = 100;
    float sum = 0;
    CvMat *tempMat = cvCreateMat(disMat->rows, disMat->cols + N - 1, disMat->type);
    for (int j = 0; j < N - 1; j++)
    {
        for (int i = 0; i < disMat->rows; i++)
        {
            CV_MAT_ELEM(*tempMat, float, i, j) = CV_MAT_ELEM(*disMat, float, i, 0);
        }
    }

    for (int j = N - 1; j < disMat->cols + N - 1; j++)
    {
        for (int i = 0; i < disMat->rows; i++)
        {
            CV_MAT_ELEM(*tempMat, float, i, j) = CV_MAT_ELEM(*disMat, float, i, j - N + 1);
        }
    }

    for (int i = 0; i < disMat->rows; i++)
    {
        for (int j = 0; j < disMat->cols; j++)
        {
            for (int k = j; k < j + N; k++)
            {
                sum += CV_MAT_ELEM(*tempMat, float, i ,k);
            }
            CV_MAT_ELEM(*disMat, float, i, j) = sum / N;
            sum = 0;
        }
    }
    cvReleaseMat(&tempMat);

    //����ӳ���match filter,��ǿλ���ź�
    double match[78] = {0,0,0,0,0,0,0.0941,0.1874,0.279,0.3682,0.454,0.5359,0.613,0.6846,0.7502,0.8091,0.8608,0.9049,0.9409,0.9686,0.9877,0.998,0.9995,
        0.9921,0.9759,0.951,0.9177,0.8762,0.827,0.7704,0.7069,0.6372,0.5619,0.4815,0.3969,0.3087,0.2179,0.125,0.0311,-0.0631,-0.1568,
        -0.249,-0.3391,-0.4261,-0.5094,-0.5881,-0.6616,-0.7292,-0.7904,-0.8446,-0.8912,-0.9299,-0.9604,-0.9824,-0.9956,-1,-0.9955,
        -0.9822,-0.9602,-0.9296,-0.8908,-0.844,-0.7898,-0.7286,-0.6609,-0.5873,-0.5085,-0.4253,-0.3382,-0.2481,-0.1558,-0.0622,0,0,0,0,0,0};

    outMat.release();
    outMat.create(disMat->rows, disMat->cols, disMat->type);
    outDataMat = outMat;

    double tmp = 0;
    for (int i = 0; i < disMat->cols; ++i)
    {
        for (int j = 0; j < disMat->rows; ++j)
        {
            for (int k = 0; k <= j; ++k)
            {
                tmp += cvmGet(disMat, k, i) * match[(j - k) < 78 ? (j - k) : 0];  /*���*/
            }
            cvmSet(&outDataMat, j, i, tmp);
            tmp = 0;
        }
    }

    cvReleaseMat(&disMat);
    cvReleaseMatHeader(&templateMat);
    cvReleaseMatHeader(&objectMat);
    cvReleaseMat(&resultMat);
}

void CDisplacement::GetMatchFilterParams(float *pData, int &num)
{
	float match[78] = {0,0,0,0,0,0,0.0941,0.1874,0.279,0.3682,0.454,0.5359,0.613,0.6846,0.7502,0.8091,0.8608,0.9049,0.9409,0.9686,0.9877,0.998,0.9995,
		0.9921,0.9759,0.951,0.9177,0.8762,0.827,0.7704,0.7069,0.6372,0.5619,0.4815,0.3969,0.3087,0.2179,0.125,0.0311,-0.0631,-0.1568,
		-0.249,-0.3391,-0.4261,-0.5094,-0.5881,-0.6616,-0.7292,-0.7904,-0.8446,-0.8912,-0.9299,-0.9604,-0.9824,-0.9956,-1,-0.9955,
		-0.9822,-0.9602,-0.9296,-0.8908,-0.844,-0.7898,-0.7286,-0.6609,-0.5873,-0.5085,-0.4253,-0.3382,-0.2481,-0.1558,-0.0622,0,0,0,0,0,0};
	num = sizeof(match) / sizeof(float);
	for (int i = 0; i < num; i++)
	{
		*pData++ = match[i];
	}
}