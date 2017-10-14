#ifndef CFILTER_H_H_H
#define CFILTER_H_H_H
#pragma   once 

#include "opencv/cv.h"
#include "CDataProcess.h"
#include <string>
#include <vector>

#define  PI 3.1415926

//////////////////////////////////////////////////////////////////////////
// same as CFilter
// use C++ class Mat
class CFilterPlus
{
public:
	CFilterPlus(const std::string&);

	void Do();

    void GetParams(float *pBuf, int &len);

private:
    void DoFilter_c();
    void DoFilter_cpp();
	void DoFilter_cuda();

	std::vector<float> param;
	int		steps;
};


#endif  //define CFILTER_H_H_H