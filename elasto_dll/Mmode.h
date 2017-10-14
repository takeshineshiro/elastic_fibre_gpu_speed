//////////////////////////////////////////////////////////////////////////
// ���紦��Ľӿ�
// ʹ��˵����
//     ����ǰ���ó�ʼ������Initialize
//     Ȼ�����DoEnvelop�����紦��
//     �����˳�ǰ����Release�ͷ���Դ
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
	// M mode �㷨ģ���ʼ����������ṹ��
	//
	//////////////////////////////////////////////////////////////////////////
	typedef struct SMModeInput
	{
		int rows; // ��ʾɨ���ߵ�����
		int cols; // ��������������/ÿ��

		int nDyn;   // ��̬��Χ
		int nEMatW; // ����ͼ����Ŀ�ȣ�
		int nEMatH; // ����ͼ����ĸ߶ȣ�

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
    // ���紦��
	//     ÿ�ε���������һ���ߵ����ݣ����а��紦��
    //////////////////////////////////////////////////////////////////////////
	extern void  DoEnvelop(const float *rf, int n, const char *file_hilber, const char *file_gray);

	//////////////////////////////////////////////////////////////////////////
	// ���紦��2
	// ���룺
	//      pmatRF������RF���ݣ� 32λfloat
	//      file_hilber�� hilber�任��Ӱ��bmp
	//      file_gray��   �Ҷ�ͼ��bmp
	//////////////////////////////////////////////////////////////////////////
	extern void  DoEnvelop2(const CvMat *pmatRF, const char *file_hilber, const char *file_gray);

	//extern void  DoEnvelop2(const CvMat *pmatRF, const char *file_hilber, CvMat *pmatGray);

	extern void  Release();
}

#ifdef __cplusplus
}
#endif 

#endif
