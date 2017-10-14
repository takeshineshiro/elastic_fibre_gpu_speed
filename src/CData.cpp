#include "stdafx.h"
#include "CData.h"
#include <fstream>
#include <iostream>


CDataset::CDataset(int rows, int cols)
{
	m_DataMat.create(rows, cols, CV_32FC1);
}

CDataset::~CDataset()
{
}

void CDataset::GetData(cv::Mat &outData)
{
	outData = m_DataMat;
}

void CDataset::GetSubData(cv::Mat &submat, int x, int y, int w, int h)
{
	
	submat = m_DataMat(cv::Rect(x,y,w,h)).clone();
}

//////////////////////////////////////////////////////////////////////////
// use Mat::at<>,  Mat::ptr<>
// 
void CDataset::ReadData(float *input)
{
	for (int i = 0; i < m_DataMat.rows; i++)
	{
		float *p = m_DataMat.ptr<float>(i);
		for (int j = 0; j < m_DataMat.cols; j++)
		{
			//m_DataMat.at<float>(i, j)) = input[i * m_DataMat->cols + j];
			*(p + j) = *(input + i * m_DataMat.cols + j);
			//p[j] = input[i * m_DataMat->cols + j];
		}
	}
}

