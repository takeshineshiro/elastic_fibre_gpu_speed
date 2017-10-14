// TestElasto.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Elasto.h"
#include "FileUtility.h"
#include <time.h>
#include <conio.h>

//#pragma comment(lib, "utility.lib")

#define  Elasto_Init        ElastoInit_gpu
#define  Elasto_Release     ElastoRelease_gpu
#define  Elasto_Process     ElastoProcess_gpu


int _tmain(int argc, _TCHAR* argv[])
{
	std::cout << "DEMO Elasto Graphy" << std::endl;

	Elasto_Init();

	// read RF data
	EInput  in;
	EOutput out;

	in.filepath_d = "displace.bmp";
	in.filepath_s = "strain.bmp";

#if 1

	// 培田采集的数据
	/*
	const char *rf_filepath = "rf.dat";
	const int  line_num = 128;
	const int  sample_num_per_line = 2048;
	*/
	const char *rf_filepath = "rf.dat";
	in.rows = 300;
	in.cols = 8192;

	in.pDatas = new float[in.rows * in.cols];
	int ok = ReadRFData(rf_filepath, in.pDatas, in.rows, in.cols);

#else

	// 原来的数据
	const char *rf_filepath = "ph1.mat";
	in.rows = 300;
	in.cols = 4000;
	in.pDatas = new float[in.rows * in.cols];
	int ok = readMatFile(rf_filepath, in.pDatas, in.rows, in.cols);

#endif 

	clock_t start, finish;
	double total;
	while (1)
	{
		printf("Please Select:\n");
		printf("\t1: Do Demo\n");
		printf("\tq or Q: quit!\n");
		bool  exit = false;
		int ch = _getche();
		switch (ch)
		{
		case '1':
			// get strain image and modulus
			start = ::clock();

			ok = Elasto_Process(in, out);

			finish = ::clock();
			total = (double)(finish - start) / CLOCKS_PER_SEC;
			printf("\nTotalTime-CPU is %fs!\n", total);
			if (ok == 0)
			{
				//std::cout << "E = " << e << " kPa" << std::endl;
				printf("\tE=%fkPa,V=%fm/s\n", out.e, out.v);

				break;
#if 0
		case '2':
			// get strain image and modulus
			start = ::clock();

			ok = ElastoProcess_gpu(in, out);

			finish = ::clock();
			total = (double) (finish - start) / CLOCKS_PER_SEC;
			printf("\nTotalTime-GPU is %fs!\n", total);
			if (ok == 0)
			{
				//std::cout << "E = " << e << " kPa" << std::endl;
				printf("\tE=%fkPa,V=%fm/s\n", out.e, out.v);
			}
			break;
#endif
		case 'q':
		case 'Q':
			exit = true;
			break;

		default:
			break;
			}
			if (exit) break;
		}

		Elasto_Release();

#if 0

		// display Strain Image 
		IplImage * image = cvLoadImage(in.filepath_d, CV_LOAD_IMAGE_UNCHANGED);
		assert(image);

		const char *window_name = "Strain Image";// 窗口的名字&标题

		cvNamedWindow(window_name);
		cvShowImage(window_name, image);
		cvWaitKey(0);
		cvDestroyWindow(window_name);
		cvReleaseImage(&image);

#else

		//printf("Press Any Key to Quit!\n");
		getchar();

#endif

		return 0;
	}


}
