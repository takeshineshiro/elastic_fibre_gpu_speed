#ifndef     ELASTO_CUDA

#define     ELASTO_CUDA


#include <stdio.h>                //   ����C������-ʵ���ϱ��������Ӧ����C�ķ�ʽ���룬�������׺Ϊcpp����
#include <stdlib.h>
#include <cuda_runtime.h>         //   ����CUDA����ʱ��ͷ�ļ�


#ifdef __cplusplus                //   ָ�������ı��뷽ʽ���Եõ�û���κ����εĺ�����


extern "C"
{
#endif

#ifdef CUDADLLTEST_EXPORTS

#define CUDADLLTEST_API __declspec(dllexport) //�������ź궨��

#else

#define CUDADLLTEST_API __declspec(dllimport)

#endif


	CUDADLLTEST_API bool initCUDA();    //Ҫ������CUDA��ʼ������



#ifdef __cplusplus
}
#endif


#endif