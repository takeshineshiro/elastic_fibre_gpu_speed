#ifndef CDATA_H_H_H
#define CDATA_H_H_H
#pragma   once 

#include "opencv\cv.h"
#include <string>

//////////////////////////////////////////////////////////////////////////
// use C++ class : cv::Mat
// same as CData
//////////////////////////////////////////////////////////////////////////
class CDataset
{
public:
	CDataset(int rows, int cols);
	~CDataset();

	void  ReadData(float *input);

	void  GetData(cv::Mat &outmat);

	void  GetSubData(cv::Mat &submat, int x, int y, int w, int h);

private:
	cv::Mat  m_DataMat;
};

#endif	//define CDATA_H_H_H