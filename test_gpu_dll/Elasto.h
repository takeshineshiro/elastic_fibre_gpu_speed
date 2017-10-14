//////////////////////////////////////////////////////////////////////////
// Export Interface of Elasto Lib  
//
//////////////////////////////////////////////////////////////////////////

#pragma once

//#ifdef ELASTO_DLL_EXPORTS
#ifndef ELASTODLL_IMPORTS
#define ELASTO_DLL_API __declspec(dllexport)
#else
#define ELASTO_DLL_API __declspec(dllimport)   
#endif

#include <iostream>

#include "SysConfig.h"


#ifdef __cplusplus
extern "C"{
#endif

	struct CvMat;

	typedef struct  SEInit
	{
		int  rows; // �������ݾ��ε���
		int  cols; // �������ݾ��ε���
	} SEInit;


	// ����
	typedef struct  EInput
	{
		float *pDatas;
		int    rows;
		int    cols;
		const char * filepath_d; // filepath of displacement image
		const char * filepath_s; // filepath of strain image

		EInput()
		{
			pDatas = 0;
			rows = -1;
			cols = -1;
			filepath_s = "";
			filepath_d = "";
		}

		~EInput()
		{
			if (pDatas)   delete [] pDatas;
		}

		void CreateDatas(int size)
		{
			if (pDatas)
			{
				delete [] pDatas;
				pDatas = 0;
			}
			pDatas = new float[size];
		}

	} EInput, *PEInput;


	// ���
	typedef struct  EOutput
	{
		float  v;  // velocity
		float  e;  // modulus
	} EOutput, *PEOutput;



	// ���Բ����¼�����
	enum EProcessEvent
	{
		EP_POST_FILTER,             // �˲����
		EP_POST_DISPLACEMENT,       // λ�Ƽ������
		EP_POST_STRAIN,             // Ӧ��������
	};


	// Elasto Prcoess Error Code
	enum
	{
		EE_OK,
		EE_FAIL,
		EE_NO_BODY,  // ����δ�Ӵ�����
	};


	typedef void (* EPHandler)(EProcessEvent, CvMat *, void *);


    //////////////////////////////////////////////////////////////////////////
    // ��ʼ����
    // ����ֵ��
    //  0�� ��ʾ�ɹ���
    //  ��0�� ��ʾʧ��
    //////////////////////////////////////////////////////////////////////////
    #define ElastoInit        ElastoInit_gpu

    #define ElastoRelease     ElastoRelease_gpu


	//////////////////////////////////////////////////////////////////////////
	// ȡ��Ӧ��ͼ�͵���ֵ
	// �㷨�⴦�������RF���ݣ����Ӧ��ͼ�͵���ֵkPa
	// Ӧ��ͼ��bmp��ʽ������file_imageָ�����ļ��У�����ģ����ֵ������e�����У�
	// ������
	//     file_image, ���룬 Ӧ��ͼ�ļ�������
	//     e��         ����� ���浯��ģ��
	//     input,      ���룬 RF���ݣ�float��ʽ
	//     rows��      ���룬 ���ݵ��У�ɨ���ߵ�����
	//     cols;       ���룬 ���ݵ��У�����һ��ɨ���ߵĲ�����
	//  ���أ�
	//     0��  �ɹ�
	//    ������ ʧ��
	//////////////////////////////////////////////////////////////////////////

    #define  ElastoProcess    ElastoProcess_gpu

	//////////////////////////////////////////////////////////////////////////
	// ע��ص�����
	// �����߿������㷨�ڲ�������̵��ض��׶εõ��ص�����Ļ���
	//////////////////////////////////////////////////////////////////////////
    #define  ElastoRegisterHandler     ElastoRegisterHandler_gpu

    #define  ElastoUnregisterHandler   ElastoUnregisterHandler_gpu

    //////////////////////////////////////////////////////////////////////////
    //  Following is Implement base GPU
    //

    //////////////////////////////////////////////////////////////////////////
    // ��ʼ����
    // ����ֵ��
    //  0�� ��ʾ�ɹ���
    //  ��0�� ��ʾʧ��
    //////////////////////////////////////////////////////////////////////////

    ELASTO_DLL_API int        ElastoInit_gpu(const char *configfile = DefaultElastoConfFile);

    ELASTO_DLL_API void       ElastoRelease_gpu();

	ELASTO_DLL_API int        ElastoProcess_gpu(const EInput &in, EOutput &out);

	ELASTO_DLL_API EPHandler  ElastoRegisterHandler_gpu(EPHandler, void *lpParam);

	ELASTO_DLL_API void       ElastoUnregisterHandler_gpu(EPHandler);


#ifdef __cplusplus
}
#endif 
