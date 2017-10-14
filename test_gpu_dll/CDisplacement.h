#ifndef CDISPLACEMENT_H_H_H
#define CDISPLACEMENT_H_H_H
#pragma   once 

#include "CDataProcess.h"
#include "opencv2/gpu/gpu.hpp"


class  CDisplacement
{
public:
    CDisplacement();
    ~CDisplacement();

    void Do();

	void GetMatchFilterParams(float *pData, int &num);

private:
	void DisplacementAlgorithm_cpp();

    void DisplacementAlgorithm_c();

	void DisplacementAlgorithm_cuda();

	void SingularFilter(cv::Mat &disp, int threshold);

	void MatchFilter(cv::Mat &disp,  cv::Mat &outDataMat);
	void MatchFilter_cuda(cv::Mat &disp,  cv::Mat &outDataMat);

	void MeanFilter(cv::Mat &disp, int steps);
	void MeanFilter_cuda(cv::Mat &disp, int steps);

	void Rel2AbsDisp(cv::Mat &disp);
	void Rel2AbsDisp_cuda(cv::Mat &disp);
};

#endif //define CDISPLACEMENT_H_H_H