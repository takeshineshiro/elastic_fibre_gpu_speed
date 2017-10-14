#include "stdafx.h"
#include "CFilter.h"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include "CElasto.h"
#include <assert.h>

#include "ElastoCudaExt.h"


void CFilterPlus::Do()
{
	{
	#if USE_CPP
		DoFilter_cpp();
	#else
		DoFilter_c();
	#endif
	}

}

void CFilterPlus::DoFilter_c()
{
    //cv::Mat &outMat = ElastoWorker::Instance().GetOutmat();
    CvMat outDataMat = ElastoWorker::Instance().GetOutmat();
    //int rows            = outDataMat.rows;
    //int cols            = outDataMat.cols;

    CvMat* tmpMat   = cvCreateMat(outDataMat.rows, outDataMat.cols, outDataMat.type);
    cvCopy(&outDataMat, tmpMat);
    float filttmp = 0.0;

    //ÕýÂË²¨
    for(int k = 0; k != outDataMat.rows; ++k)
    {
        for (int i = 0; i != steps - 1; ++i)
        {
            filttmp = 0.0;
            for (int j = 0; j <= i; ++j)
            {
                filttmp += param[j] * (*(static_cast<float*>(static_cast<void*>(CV_MAT_ELEM_PTR(*tmpMat, k, i - j)))));
            }
            for (int j=i+1; j<steps-1;j++)
            {
                filttmp += param[j] *(*(static_cast<float*>(static_cast<void*>(CV_MAT_ELEM_PTR(*tmpMat, k, 0)))));
            }
            *(static_cast<float*>(static_cast<void*>(CV_MAT_ELEM_PTR(outDataMat, k, i)))) = filttmp;
        }

        for (int i = steps - 1; i != outDataMat.cols; ++i)
        {
            filttmp = 0.0;
            for (std::vector<float>::size_type ix = 0; ix != steps - 1; ++ix)
            {
                filttmp += param[ix] * (*(static_cast<float*>(static_cast<void*>(CV_MAT_ELEM_PTR(*tmpMat, k, i - ix)))));
            }
            *(static_cast<float*>(static_cast<void*>(CV_MAT_ELEM_PTR(outDataMat, k, i)))) = filttmp;
        }
    }

    //ÄæÂË²¨
    cvCopy(&outDataMat, tmpMat);
    for (int k = 0; k != outDataMat.rows; ++k)
    {
        for (int i = steps - 1; i != outDataMat.cols; ++i)
        {
            filttmp = 0.0;
            for (std::vector<double>::size_type ix = 0; ix != param.size(); ++ix)
            {
                filttmp += param[ix] * (*(static_cast<float*>(static_cast<void*>(CV_MAT_ELEM_PTR(*tmpMat, k, outDataMat.cols - i - 1 + ix)))));
            }
            *(static_cast<float*>(static_cast<void*>(CV_MAT_ELEM_PTR(outDataMat, k, outDataMat.cols - i - 1)))) = filttmp;
        }
    }

    cvReleaseMat(&tmpMat);
}


CFilterPlus::CFilterPlus(const std::string &filename)
{
    if (filename.size() == 0)
    {
        return;
    }

    std::fstream paramFile(filename.c_str());
    if (!paramFile)
    {
        return;
    }

    std::stringstream ss;
    float tmp;
    std::string str;

    param.clear();
    while (!paramFile.eof())
    {
        ss.clear();
        paramFile >> str;
        ss << str;
        ss >> tmp;
        param.push_back(tmp);
    }
    steps = param.size();
    paramFile.close();
    ss.clear();
}


void CFilterPlus::GetParams(float *pBuf, int &len)
{
    assert(len >= steps);
    len = steps;

    int i;
    for (i = 0; i < steps; i++)
    {
        pBuf[i] = param[i];
    }
}

void CFilterPlus::DoFilter_cpp()
{
	cv::Mat &outMat = ElastoWorker::Instance().GetOutmat();
	cv::Mat tmpMat  = outMat.clone();
	int rows        = outMat.rows;
	int cols        = outMat.cols;

	float filttmp = 0.0;
    float *p;

	//ÕýÂË²¨
	for(int k = 0; k != rows; ++k)
	{
		for (int i = 0; i != steps - 1; ++i)
		{
			filttmp = 0.0;
			for (int j = 0; j <= i; ++j)
			{
                p = tmpMat.ptr<float>(k, i - j);
				filttmp += param[j] * (*p);
			}
			for (int j = i + 1; j < steps - 1; j++)
			{
                p = tmpMat.ptr<float>(k, 0);
				filttmp += param[j] * (*p);
			}
			*outMat.ptr<float>(k, i) = filttmp;
		}

		for (int i = steps - 1; i != cols; ++i)
		{
			filttmp = 0.0;
			for (std::vector<float>::size_type ix = 0; ix != steps - 1; ++ix)
			{
				filttmp += param[ix] * (*tmpMat.ptr<float>(k, i - ix));
			}
			*outMat.ptr<float>(k, i) = filttmp;
		}
	}

	//ÄæÂË²¨
    outMat.copyTo(tmpMat);
	
	for (int k = 0; k != rows; ++k)
	{
		for (int i = steps - 1; i != cols; ++i)
		{
			filttmp = 0.0;
			for (std::vector<double>::size_type ix = 0; ix != param.size(); ++ix)
			{
                p = tmpMat.ptr<float>(k,  cols - i - 1 + ix);
				filttmp += param[ix] * (*p);
			}
			*outMat.ptr<float>(k, cols - i - 1) = filttmp;
		}
	}
}

void CFilterPlus::DoFilter_cuda()
{
	cv::Mat &outMat = ElastoWorker::Instance().GetOutmat();

	elasto_cuda::Wrapper::Instance().BpFilter(outMat);
}