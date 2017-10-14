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
//  按照公式，自己做的实现；每次循环，都计算模板templt每个像素的平方值，以及对象矩阵image内部某个w×h矩阵的像素平方值
//  模板的计算每次都重复了。同时
//  由于每次只移动step, 在image（对象）矩阵中有许多元素在每次循环重复计算；
//  功能都正确，但是计算耗时比cvMatchTemplate多很多
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
// 在min_max_loc_32f基础上简化，直接计算。
// 经调试验证： 功能正确。
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
	//待删除 2009-10-29 鲁棒性不够
	for (int i = 1; i < dispMat.rows - 1; ++i)   //位移平滑，去奇异点，以互相关系数为依据。
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
	//新添加程序，match filter,增强位移信号
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
				tmp += (*dispMat.ptr<float>(k, i)) * match[(j - k) < 78 ? (j - k) : 0];  /*卷积*/
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
    int    multiWin = 2;   //大窗口对小窗口的倍数

    int    winSize  = configParam.windowHW;
    int    stepSize = configParam.step;

    int     winNum      = (outDataMat.cols - multiWin * winSize) / stepSize;  //计算需要匹配的段数
    cv::Mat dispMat(outDataMat.rows - 1, winNum, CV_32FC1);     //用于保存位移的矩阵，匹配得出的整数位移经过插值后应该为浮点数
    cv::Mat templateMat; //= cvCreateMatHeader(1, winSize, CV_32FC1);  //用于匹配的模板 
    cv::Mat objectMat;   //= cvCreateMatHeader(1, multiWin * winSize, CV_32FC1);  //用于匹配的目标
    cv::Mat resultMat;  //用于存放匹配结果

    double    min, max;     //匹配结果中的最大值
    cv::Point max_loc;      //匹配结果最大值所在位置
    double    pmax, nmax;   //最大值的前后两个值
    double    displacement; //存储位移值

    //#pragma  omp  parallel for schedule(dynamic)
    float *p;

    for (int i = 0; i < dispMat.rows; ++i)   //求位移
    {
        for (int j = 0; j < dispMat.cols; ++j)
        {   
            templateMat = outDataMat(cv::Rect((multiWin - 1) * winSize / 2 + j * stepSize, i, winSize, 1));
            objectMat = outDataMat(cv::Rect(j * stepSize, i + 1, multiWin * winSize, 1));

            cv::matchTemplate(objectMat, templateMat, resultMat, CV_TM_CCORR_NORMED); //匹配

            cv::minMaxLoc(resultMat, &min, &max, NULL, &max_loc);
			/*
            {
			    // 调用自己写的算法进行计算，它应该和opencv的结果相同
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

            displacement = (multiWin - 1) * winSize / 2 - max_loc.x - (pmax - nmax) / (2 * (pmax - 2 * max + nmax));   //插值得到位移
            p = dispMat.ptr<float>(i, j);
            *p = static_cast<float>(displacement); //结果存放到位移矩阵中
        }
    }

    //位移平滑，去奇异点，以互相关系数为依据。
	SingularFilter(dispMat, 12);

    //位移叠加
	Rel2AbsDisp(dispMat);
	//(ElastoWorker::Instance().CudaIsCapable()) ? Rel2AbsDisp_cuda(dispMat) : Rel2AbsDisp(dispMat);

	//均值滤波
	//(ElastoWorker::Instance().CudaIsCapable()) ? MeanFilter_cuda(dispMat, 100) : MeanFilter(dispMat, 100);
	MeanFilter(dispMat, 100);

    //match filter,增强位移信号
	MatchFilter(dispMat, outDataMat);
	//(ElastoWorker::Instance().CudaIsCapable()) ? MatchFilter_cuda(dispMat, outDataMat) : MatchFilter(dispMat, outDataMat);
}

void CDisplacement::DisplacementAlgorithm_cuda()
{
    cv::Mat &outDataMat      = ElastoWorker::Instance().GetOutmat();
    ConfigParam &configParam = ElastoWorker::Instance().GetConfigParam();
    int    multiWin = 2;   //大窗口对小窗口的倍数

    int    winSize  = configParam.windowHW;
    int    stepSize = configParam.step;

    int     winNum      = (outDataMat.cols - multiWin * winSize) / stepSize;  //计算需要匹配的段数
	cv::Mat dispMat;
	elasto_cuda::Wrapper::Instance().CalcDisplacement(outDataMat, dispMat, winSize, stepSize, multiWin);

    //鲁棒性不够
    //位移平滑，去奇异点，以互相关系数为依据。
	SingularFilter(outDataMat, 12);

    //位移叠加
	Rel2AbsDisp_cuda(dispMat);

    //均值滤波
    MeanFilter_cuda(dispMat, 100);

    //match filter,增强位移信号
	MatchFilter_cuda(dispMat, outDataMat);
}

//////////////////////////////////////////////////////////////////////////
// opencv的C接口函数有许多不方便的地方，且容易造成内存泄漏。
// 下面的代码templateMat, objectMat, 我改成了cvCreateMatHeader, 应该更合理。
// 原来的代码似乎有可能内存泄露， data.ptr指针的值已经改写了。奇怪的是release时却没有报错
// 看来，opencv内部代码在防御设计上考虑不周，还是我没有足够理解opencv的设计
//     杨戈
//////////////////////////////////////////////////////////////////////////
void CDisplacement::DisplacementAlgorithm_c()
{
    cv::Mat &outMat          = ElastoWorker::Instance().GetOutmat();
    CvMat  outDataMat        = outMat;
    ConfigParam &configParam = ElastoWorker::Instance().GetConfigParam();

    int    winSize  = configParam.windowHW;
    int    stepSize = configParam.step;

    int    multiWin = 2;   //大窗口对小窗口的倍数
    int    winNum   = static_cast<int>(outDataMat.cols - multiWin * winSize) / stepSize;  //计算需要匹配的段数
    CvMat *disMat      = cvCreateMat(outDataMat.rows - 1, winNum, CV_32FC1);     //用于保存位移的矩阵，匹配得出的整数位移经过插值后应该为浮点数
    CvMat *templateMat = cvCreateMatHeader(1, winSize, CV_32FC1);  //用于匹配的模板 
    CvMat *objectMat   = cvCreateMatHeader(1, multiWin * winSize, CV_32FC1);  //用于匹配的目标
    CvMat *resultMat   = cvCreateMat(1, (multiWin - 1) * winSize + 1, CV_32FC1);  //用于存放匹配结果

    //CvMat *maxMat      = cvCreateMat(outDataMat->rows - 1, winNum, CV_32FC1);    //用于存储每次匹配的最大匹配结果

    double min, max;  //匹配结果中的最大值
    CvPoint max_loc;  //匹配结果最大值所在位置
    double pmax, nmax; //最大值的前后两个值
    double displacement; //存储位移值
    //#pragma  omp  parallel for schedule(dynamic)

    for (int i = 0; i < disMat->rows; ++i)   //求位移
    {
        for (int j = 0; j < disMat->cols; ++j)
        {         /* ->data.ptr表示数据的起始位置 */
            templateMat->data.ptr = outDataMat.data.ptr + i * outDataMat.step 
                + static_cast<int>(sizeof(float) * ((multiWin - 1) * winSize / 2 + j * stepSize)); 
            objectMat->data.ptr   = outDataMat.data.ptr + (i + 1) * outDataMat.step + static_cast<int>(sizeof(float) * (j * stepSize)/* + 0.5*/);
            cvMatchTemplate(objectMat, templateMat, resultMat, CV_TM_CCORR_NORMED); //匹配
			//match_template_tm_ccorr_normed_32f(cv::Mat(objectMat), cv::Mat(templateMat), cv::Mat(resultMat));
            cvMinMaxLoc(resultMat, &min, &max, NULL, &max_loc);
            //pmax = *static_cast<float*>(static_cast<void*>(resultMat->data.ptr + sizeof(float)*max_loc.x - sizeof(float)));  //取最大值的前一个值
            //nmax = *static_cast<float*>(static_cast<void*>(resultMat->data.ptr + sizeof(float)*max_loc.x + sizeof(float)));  //取最大值的后一个值
			pmax = *static_cast<float*>(static_cast<void*>(resultMat->data.ptr 
				                                           + sizeof(float) * ((max_loc.x - 1) < 0 ? 0 : max_loc.x - 1)));
			nmax = *static_cast<float*>(static_cast<void*>(resultMat->data.ptr 
				                                           + sizeof(float) * ((max_loc.x + 1) < resultMat->cols ? max_loc.x + 1 : max_loc.x)));  //取最大值的后一个值

            displacement = (multiWin - 1) * winSize / 2 - max_loc.x - (pmax - nmax) / (2 * (pmax - 2 * max + nmax));   //插值得到位移

            CV_MAT_ELEM(*disMat,float, i, j) = static_cast<float>(displacement); //结果存放到位移矩阵中
        }
    }

    //待删除 2009-10-29 鲁棒性不够
    for (int i = 1; i < disMat->rows - 1; ++i)   //位移平滑，去奇异点，以互相关系数为依据。
    {
        for (int j = 0; j < disMat->cols - 1; ++j)
        {
            if (abs(CV_MAT_ELEM(*disMat, float, i, j)) > 12)
            {
                CV_MAT_ELEM(*disMat, float, i, j) = CV_MAT_ELEM(*disMat, float, i - 1, j);  
            }
        }
    }

    //位移叠加
    for (int i = 0; i < disMat->cols; ++i)
    {
        for(int j = 0; j < disMat->rows - 1; ++j)
        {
            CV_MAT_ELEM(*disMat, float, j + 1, i) = CV_MAT_ELEM(*disMat, float, j + 1, i) + CV_MAT_ELEM(*disMat, float, j, i);
        }
    }

    //均值滤波
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

    //新添加程序，match filter,增强位移信号
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
                tmp += cvmGet(disMat, k, i) * match[(j - k) < 78 ? (j - k) : 0];  /*卷积*/
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