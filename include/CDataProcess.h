#ifndef CDATAPROCESS_H_H_H
#define CDATAPROCESS_H_H_H
#pragma   once 


//#include "opencv\cv.h"
struct CvMat;

class CDataProcess{
public:
	CDataProcess();
	CvMat* doProcess(CDataProcess*);
	virtual void Do();
public:
	//static void Create(int rows, int cols);
	//static void Release();

	static CvMat *inDataMat;
	static CvMat *outDataMat;
};
#endif //define CDATAPROCESS_H_H_H