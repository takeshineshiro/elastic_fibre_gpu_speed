#ifndef CSTRAIN_H_H_H
#define CSTRAIN_H_H_H
#pragma   once 

#include "opencv\cv.h"
#include "CDataProcess.h"
#include <iostream>

//struct CvMat;
struct CvPoint;
struct EInput;
struct EOutput;

struct MyLessThan
{
	bool operator() (CvPoint &lhs, CvPoint &rhs)
	{
		return (lhs.x - lhs.y) < (rhs.x - rhs.y);
	}

    bool operator() (cv::Point &lhs, cv::Point &rhs)
    {
        return (lhs.x - lhs.y) < (rhs.x - rhs.y);
    }
};

class CStrainPlus
{
public:
    // 多次拉东变换
    void  CalcStrain(const EInput &input, EOutput &output);

	void  CalcStrain_cuda(const EInput &input, EOutput &output);

    //////////////////////////////////////////////////////////////////////////
    // 计算应变值和应变图的灰度值
    // 输入：
    //    count， 拟合的点数
    //    pmat，  应变的矩阵
    //    pimg，  应变图
    //////////////////////////////////////////////////////////////////////////
    void  ComputeStrainValueAndImage(const cv::Mat &in, int count, cv::Mat &mat, cv::Mat &img);

    void  ComputeStrainValueAndImage(int count, cv::Mat &mat, cv::Mat &img);

    void  RadonProcess_c(CvPoint &s, CvPoint &e, const CRect &rect, const CvMat &matStrain);

    void  RadonProcess_cpp(cv::Point &s, cv::Point &e, const cv::Rect &rect, const cv::Mat &matStrain);

    void  RadonProcess_cuda(cv::Point &s, cv::Point &e, const cv::Rect &rect, const cv::Mat &matStrain);

};

void  RadonSum(const cv::Mat &inMat, cv::Mat &radonMat);

void  RadonSum(const CvMat *pmatDisplacement, CvMat **ppmatRodan);


#endif //define CSTRAIN_H_H_H