//////////////////////////////////////////////////////////////////////////
// 包络处理的接口
// 使用说明：
//     调用前调用初始化函数Initialize
//     然后调用DoEnvelop做包络处理，
//     程序退出前调用Release释放资源
//////////////////////////////////////////////////////////////////////////

#ifndef INTERFACE_H_H_H
#define INTERFACE_H_H_H

#include <iostream>

struct CvMat;

#ifdef __cplusplus
extern "C"{
#endif

namespace mmode
{
	//////////////////////////////////////////////////////////////////////////
	// M mode 算法模块初始化输入参数结构体
	//
	//////////////////////////////////////////////////////////////////////////
	typedef struct SMModeInput
	{
		int rows; // 表示扫描线的数量
		int cols; // 数据样本的数量/每线

		int nDyn;   // 动态范围
		int nEMatW; // 包络图矩阵的宽度，
		int nEMatH; // 包络图矩阵的高度，

		SMModeInput()
		{
			rows = 0;
			cols = 0;
			nDyn = 0;
			nEMatW = 0;
			nEMatH = 0;
		}

	} SMModeInput;

	extern void Initialize(const SMModeInput &);

	//////////////////////////////////////////////////////////////////////////
    // 包络处理
	//     每次调用者输入一条线的数据，进行包络处理
    //////////////////////////////////////////////////////////////////////////
	extern void  DoEnvelop(const float *rf, int n, const char *file_hilber, const char *file_gray);

	//////////////////////////////////////////////////////////////////////////
	// 包络处理2
	// 输入：
	//      pmatRF，整幅RF数据， 32位float
	//      file_hilber， hilber变换的影像，bmp
	//      file_gray，   灰度图，bmp
	//////////////////////////////////////////////////////////////////////////
	extern void  DoEnvelop2(const CvMat *pmatRF, const char *file_hilber, const char *file_gray);

	//extern void  DoEnvelop2(const CvMat *pmatRF, const char *file_hilber, CvMat *pmatGray);

	extern void  Release();
}

#ifdef __cplusplus
}
#endif 

#endif
