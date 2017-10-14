//////////////////////////////////////////////////////////////////////////
//
//
//////////////////////////////////////////////////////////////////////////
#pragma warning( disable : 4819 )

#define ELASTO_CUDA_EX_CLASS  __declspec(dllexport)

#define ELASTO_CUDA_API       __declspec(dllexport)


#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "devMatchCuda.cuh" 
#include <opencv2\opencv.hpp>
#include <opencv2\gpu\gpu.hpp>
#include <opencv2/gpu/device/common.hpp>
#include "CudaComputer.cuh"

using namespace cv;

using namespace cv::gpu;

using namespace cv::gpu::device;

namespace elasto_cuda
{

    __global__ void  calcStrainKernal(PtrStepSzf in, int count, PtrStepSzf out); //计算应变


    __device__ void  fitLine2D_wods(float2 * points, int _count, float *line);  // 最小二乘法做直线拟合


    __global__ void  filterKernal_front(PtrStepSzf in, PtrStepSzf params, PtrStepSzf out); // 正滤波


    __global__ void  filterKernal_back(PtrStepSzf in, PtrStepSzf params, PtrStepSzf out);  // 逆滤波


    __device__ void  minMaxLoc_32f(PtrStepSzf in, float *min_val, float *max_val, CvPoint *min_loc, CvPoint *max_loc); 


    __device__ void  matchTemplate_tm_ccorr_normed_32f(PtrStepSzf image, PtrStepSzf templt, PtrStepSzf result);


    __global__ void  calcDispKernal(PtrStepSzf inData, PtrStepSzf dispMat, PtrStepSz<PtrElemType> resultMats, int win, int step, int scale);



    template<typename T>   PtrStepSz<T>  make_PtrStepSz(void *data, int rows, int cols, int step)
    {

        return PtrStepSz<T>(rows, cols, (T*)data, step);

    }


    template<typename T>   PtrStep<T>   make_PtrStep(void *data, int step)
    {

        return PtrStep<T>((T*)data, step);
    }


    int  TotalBytes(const Size &size)
    {

        return (size.width * size.height * sizeof(ElemType));

    }


    int  TotalBytes(const PtrStepSzf &ptr)
    {
        //return ptr.cols * ptr.rows * sizeof(float);
        return  ptr.rows * ptr.step;

    }


    int  TotalBytes(const cv::Mat &mat)
    {

        return mat.elemSize() * mat.total();

    }


    int  TotalBytes(const CvMat &mat)

    {

        return mat.rows * mat.step;

    }


	//////////////////////////////////////////////////////////////////////////
	// 直线拟合-最小二乘法
	//////////////////////////////////////////////////////////////////////////


    __device__ void fitLine2D_wods(float2 *points, int _count, float *line)
    {

        float x = 0, y = 0, x2 = 0, y2 = 0, xy = 0, w = 0;

        float dx2, dy2, dxy;

        int i;

        int count = _count;

        float t;


        /* Calculating the average of x and y... */
        {
            for (i = 0; i < count; i += 1)
            {
                x += points[i].x;
                y += points[i].y;
                x2 += points[i].x * points[i].x;
                y2 += points[i].y * points[i].y;
                xy += points[i].x * points[i].y;
            }
            w = (float) count;
        }

        x /= w;
        y /= w;
        x2 /= w;
        y2 /= w;
        xy /= w;

        dx2 = x2 - x * x;
        dy2 = y2 - y * y;
        dxy = xy - x * y;

        t = (float) atan2f( 2 * dxy, dx2 - dy2 ) / 2;
        line[0] = (float) cosf( t );
        line[1] = (float) sinf( t );

        line[2] = (float) x;
        line[3] = (float) y;

    }


	//////////////////////////////////////////////////////////////////////////
	// 计算应变
	//////////////////////////////////////////////////////////////////////////

    __global__ void  calcStrainKernal(PtrStepSzf in, int count, PtrStepSzf out)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;// index of Mat in

        int idy = blockIdx.y * blockDim.y + threadIdx.y;// index of Mat in


        if (idx < in.cols && idx >= count - 1 && idy < in.rows)
        {

			float  coeff; //存储直线斜率（一次项）

			int    deltadepth = 5;   //单位深度

			float  result[4] = {0.0, 0.0, 0.0, 0.0}; //存储直线拟合结果

			int    i;

			float2 pts[32];

            for (i = 0; i < count; i++)
            {

                pts[i].x = (idx + i - count - 1) * deltadepth;

                pts[i].y = in(idy, idx + i - count - 1);

            }

            fitLine2D_wods(pts, count, result);//最小二乘拟合

            coeff = result[1] / result[0];     //算出直线斜率，即为中心点应变

			out(idy, idx - count - 1) = coeff;

        }

    }


	//////////////////////////////////////////////////////////////////////////
	//  拉东变换
	//////////////////////////////////////////////////////////////////////////
    __global__ void  radon_kernal(PtrStepSzf in, PtrStepSzf out)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;// index of Mat Out, cols


        int idy = blockIdx.y * blockDim.y + threadIdx.y;// index of Mat Out, rows


        if (idx < out.cols  && idy < out.rows)
        {

			int xstart = 0;

			int xend   = in.rows;

			int   dx = 0;

			float dt = 0.0f;

			float c = 0.0f;

            out(idy, idx) = 0.0f;

            if (idx > idy)
            {
                c = (float)(idx - idy) / (xend - xstart);

                for (dx = xstart; dx < xend; dx++)
                {
                    dt = idy + (dx - xstart) * c;

					out(idy, idx) = out(idy, idx) + in(dx, (int)dt);

                }
            }
        }
    }




    // 正滤波
    __global__ void  filterKernal_front(PtrStepSzf in, PtrStepSzf params, PtrStepSzf out)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;// index of Mat in


        int idy = blockIdx.y * blockDim.y + threadIdx.y;// index of Mat in


        float sum = 0.0f;


        if (idx < in.cols && idy < in.rows)
        {

            int    i;

            float *pIn     = in.ptr(idy);

            float *pOut    = out.ptr(idy);

            float *pParams = params.ptr(0);

            if (idx < params.cols - 1)
            {

                // 下面的代码我觉得有一些问题， 计算的次数只有用step-1, 而不是step；step是param的长度
                for (i = 0; i <= idx; i++)
                {

                    sum += pParams[i] * pIn[idx - i];

                }

                for (i = idx + 1; i < params.cols - 1; i++)
                {
                    sum += pParams[i] * pIn[0];
                }
                pOut[idx] = sum;
            }
            else
            {
                for (i = 0; i < params.cols - 1; i++)
                {
                    sum += pParams[i] * pIn[idx - i];
                }
                pOut[idx] = sum;
            }
        }
    }



    // 逆滤波
    __global__ void  filterKernal_back(PtrStepSzf in, PtrStepSzf params, PtrStepSzf out)
    {
      
		int idx = blockIdx.x * blockDim.x + threadIdx.x;// index of Mat in

        int idy = blockIdx.y * blockDim.y + threadIdx.y;// index of Mat in

        float sum = 0.0f;

        if (idx < in.cols && idy < in.rows)
        {

            int    i;
            float *pIn     = in.ptr(idy);
            float *pOut    = out.ptr(idy);
            float *pParams = params.ptr();
            // 下面代码， in的每一行中，最后step个数据没有进行计算。这与正滤波的处理不同
            if (idx < in.cols - params.cols)
            {
                for (i = 0; i < params.cols; i++)
                {
                    sum += pParams[i] * pIn[idx + i];
                }
                pOut[idx] = sum;
            }
        }
    }



	//////////////////////////////////////////////////////////////////////////
	// 计算矩阵的最大最小值，并输出其在矩阵中的位置
	//
	//////////////////////////////////////////////////////////////////////////


    __device__ void  minMaxLoc_32f(PtrStepSzf img, float *min_val, float *max_val, CvPoint *min_loc, CvPoint *max_loc)
    {

        float  fminval = FLT_MAX;

        float  fmaxval = -fminval;

        int  rows = img.rows;

        int  cols = img.cols;

        float val;

        CvPoint min_idx, max_idx;

        int i, j;

        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                val = img(i, j);
                if (val > fmaxval)
                {
                    fmaxval = val;
                    max_idx.x = j;
                    max_idx.y = i;
                }
                if (val < fminval)
                {
                    fminval = val;
                    min_idx.x = j;
                    min_idx.y = i;
                }
            }
        }

        if (min_val)
        {
            *min_val = fminval;
        }
        if (max_val)
        {
            *max_val = fmaxval;
        }
        if (min_loc)
        {
            *min_loc = min_idx;
        }
        if (max_loc)
        {
            *max_loc = max_idx;
        }

    }


	//////////////////////////////////////////////////////////////////////////
	// 模板匹配的内核函数
	//    数据类型是float；
	//    数据矩阵都是基于step存取的。
	//    image， 对象矩阵
	//    templt, 模板
	//    result, 保存结果
	//////////////////////////////////////////////////////////////////////////

    __device__ void  matchTemplate_tm_ccorr_normed_32f(PtrStepSzf image, PtrStepSzf templt, PtrStepSzf result)
    {
        int i, j, w, h;

        int rows = image.rows - templt.rows + 1;

        int cols = image.cols - templt.cols + 1;

        float  sum       = 0.0f;

        float  img_sum   = 0.0f;

        float  tmpl_sum  = 0.0f;

        for (i = 0; i < rows; i++)
        {
            for (j = 0; j < cols; j++)
            {
                sum      = 0.0f;

                img_sum  = 0.0f;

                tmpl_sum = 0.0f;
                for (h = 0; h < templt.rows; h++)
                {
                    for (w = 0; w < templt.cols; w++)
                    {

                        sum += image(i + h, j + w) * templt(h, w);

                        img_sum  += image(i + h, j + w) * image(i + h, j + w);

                        tmpl_sum += templt(h, w) * templt(h, w);

                    }
                }

                result(i, j) = sum / ::sqrtf(img_sum * tmpl_sum);

            }
        }
    }



    __global__ void  calcDispKernal(PtrStepSzf inData, PtrStepSzf dispMat, PtrStepSz<PtrElemType> resultMats, int win, int step, int scale)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x; // cols

        int idy = blockIdx.y * blockDim.y + threadIdx.y; // rows


		if (idx < dispMat.cols && idy < dispMat.rows)
		{
			int len = (scale - 1) * win + 1;

			PtrStepSzf  image  = PtrStepSz<float>(1, win * scale, inData.ptr(idy + 1) + idx * step, win * scale * sizeof(ElemType));
			
			PtrStepSzf  templt = PtrStepSz<float>(1, win, inData.ptr(idy) + idx * step + (scale - 1) * win / 2, win * sizeof(ElemType));
			
			PtrStepSzf  result = PtrStepSz<float>(1, len, resultMats(idy, idx), len * sizeof(ElemType));
			
			matchTemplate_tm_ccorr_normed_32f(image, templt, result);


			float   min_val, max_val;

			CvPoint max_loc;

			minMaxLoc_32f(result, &min_val, &max_val, 0, &max_loc);

			float pmax = result(0, (max_loc.x - 1) < 0 ? 0 : (max_loc.x - 1));

			float nmax = result(0, (max_loc.x + 1) < result.cols ? (max_loc.x + 1) : max_loc.x);


			float displacement = (scale - 1) * win / 2 - max_loc.x - (pmax - nmax) / (2 * (pmax - 2 * max_val + nmax));//插值得到位移
			
			dispMat(idy, idx)  = displacement; //结果存放到位移矩阵中

		}
    }




	__global__  void  MeanFilter_Kernal(PtrStepSzf in, PtrStepSzf out, int step)
	{
		
		int idx = blockIdx.x * blockDim.x + threadIdx.x; // cols

		int idy = blockIdx.y * blockDim.y + threadIdx.y; // rows


		if (idx < out.cols && idy < out.rows)
		{

			float sum = 0.0f;

			for (int i = 0; i < step; i++)
			{

				sum += in(idy, idx + i);

			}

			out(idy, idx) = sum / step;

		}

	}




	__global__  void  MatchFilter_Kernal(PtrStepSzf clone, PtrStepSzf disp, PtrStepSzf params)
	{

		int idx = blockIdx.x * blockDim.x + threadIdx.x; // cols

		int idy = blockIdx.y * blockDim.y + threadIdx.y; // rows

		if (idx < disp.cols && idy < disp.rows)
		{

			int i;

			float sum = 0.0f;

			for (i = 0; i < idy + 1; i++)// row
			{

				sum += clone(i, idx) * params(0, (idy - i) < params.cols ? (idy - i) : 0);

			}

			disp(idy, idx) = sum;

		}

	}




	__global__  void  Rel2Abs_Kernal(PtrStepSzf in, PtrStepSzf out)
	{

		int idx = blockIdx.x * blockDim.x + threadIdx.x; // cols

		int idy = blockIdx.y * blockDim.y + threadIdx.y; // rows


		if (idx < out.cols && idy < out.rows)
		{

			float sum = 0.0f;

			for (int i = 0; i < idy + 1; i++)
			{
				sum += in(i, idx);
			}

			out(idy, idx) = sum;

		}
	}



   


    CudaComputer::CudaComputer():m_fAvaible(false)
    {

    }


    CudaComputer::~CudaComputer()
    {

        if (m_fAvaible)
        {

            FreeHostMem();

            FreeCudaMem();


            cudaDeviceReset();

        }

    }


    int CudaComputer::Init(const ConfigParam &param)
    {

        // check the compute capability of the device

        int num_devices=0;


        cudaGetDeviceCount(&num_devices);

        if (0 == num_devices)
        {
            printf("your system does not have a CUDA capable device, waiving test...\n");
            return ERR_NO_GPU;
        }




		if (initCUDA())   {
		
		
			m_fAvaible = true;
		
		
		}


    //    m_fAvaible = true;

        AllocHostMem();

        AllocCudaMem();

        return ERR_OK;


    }





    void CudaComputer::InitBandpassFilterParams(float *pDatas, int num)
    {

		void *p;

		cudaMalloc(&p, sizeof(ElemType) * num);

		cudaMemcpy(p, pDatas, sizeof(ElemType) * num, cudaMemcpyHostToDevice);

		m_ptrBpParams_d = make_PtrStepSz<ElemType>(p, 1, num, sizeof(ElemType) * num);

    }






    void CudaComputer::InitLowpassFilterParams(float *pDatas, int num)
    {

		void *p;

		cudaMalloc(&p, sizeof(ElemType) * num);

		cudaMemcpy(p, pDatas, sizeof(ElemType) * num, cudaMemcpyHostToDevice);

		m_ptrLpParams_d = make_PtrStepSz<ElemType>(p, 1, num, sizeof(ElemType) * num);

    }





	void CudaComputer::InitMatchFilterParams(float *pDatas, int num)
	{
		void *p;


		cudaMalloc(&p, sizeof(ElemType) * num);

		cudaMemcpy(p, pDatas, sizeof(ElemType) * num, cudaMemcpyHostToDevice);

		m_ptrMatchParams_d = make_PtrStepSz<ElemType>(p, 1, num, sizeof(ElemType) * num);

	}






    void CudaComputer::AllocHostMem()
    {
        void *p;

        cv::Size size;

		size = Wrapper::Instance().GetMatSize("MOI");

		cudaHostAlloc(&p, TotalBytes(size), 0);

		m_ptrInFrame = make_PtrStepSz<ElemType>(p, size.height, size.width, size.width * sizeof(ElemType));

		size = Wrapper::Instance().GetMatSize("StrainMat");

        cudaHostAlloc(&p, TotalBytes(size), 0);

        m_ptrStrainMat = make_PtrStepSz<ElemType>(p, size.height, size.width, size.width * sizeof(ElemType));

        size = Wrapper::Instance().GetMatSize("DispMat");

        cudaHostAlloc(&p, TotalBytes(size), 0);

        m_ptrDispMat = make_PtrStepSz<ElemType>(p, size.height, size.width, size.width * sizeof(ElemType));

    }





    void CudaComputer::FreeHostMem()
    {

        if (m_ptrStrainMat.data)
        {
            cudaFreeHost(m_ptrStrainMat.data);
        }

        if (m_ptrDispMat.data)
        {
            cudaFreeHost(m_ptrDispMat.data);
        }

		if (m_ptrInFrame.data)
		{
			cudaFreeHost(m_ptrInFrame.data);
		}

    }





    void CudaComputer::AllocCudaMem()
    {
        void *p;

        cv::Size size = Wrapper::Instance().GetMatSize("StrainMat");

        cudaMalloc(&p, TotalBytes(size));

        m_ptrStrainMat_d = make_PtrStepSz<ElemType>(p, size.height, size.width, size.width * sizeof(ElemType));

        size = Wrapper::Instance().GetMatSize("DispMat");

        cudaMalloc(&p, TotalBytes(size));

        m_ptrDispMat_d = make_PtrStepSz<ElemType>(p, size.height, size.width, size.width * sizeof(ElemType));

        size = Wrapper::Instance().GetMatSize("MOI");

        cudaMalloc(&p, TotalBytes(size));

        m_ptrInFrame_d = make_PtrStepSz<ElemType>(p, size.height, size.width, size.width * sizeof(ElemType));

        size = Wrapper::Instance().GetMatSize("MOI");

        cudaMalloc(&p, TotalBytes(size));

        m_ptrFilterBuf_d = make_PtrStepSz<ElemType>(p, size.height, size.width, size.width * sizeof(ElemType));
		
        {
			// 首先在host创建一块内存，在host创建device的内存并把地址保存在pTmp中
			// 设计pMat是只是为了方便操作

			PtrElemType *pTmp = new PtrElemType[size.width * size.height];

			PtrStepSz<PtrElemType>  pMats = make_PtrStepSz<PtrElemType>((void*)pTmp, size.height, size.width, size.width * sizeof(PtrElemType));

            int c, r;

            cv::Size  size_mat = Wrapper::Instance().GetMatSize("ResultMat");


			// 在device上创建disp计算-模板匹配处理中保存结果的矩阵数据区，然后保存在pTmp指向的存储区。注意，pTmp在host端
            for (r = 0; r < size.height; r++)
            {
                for (c = 0; c < size.width; c++)
                {

                    cudaMalloc(&p, sizeof(ElemType) * size_mat.height * size_mat.width);


                    pMats(r, c) = (PtrElemType)p;


                }

            }


			// 在device创建一个和pMats等大小的内存区，然后把pMats指向的数据区的内容拷贝到m_ptrResultMats_d中
		    size = Wrapper::Instance().GetMatSize("ResultMats");


            cudaMalloc(&p, sizeof(PtrElemType) * size.height * size.width);


            m_ptrResultMats_d = make_PtrStepSz<PtrElemType>(p, size.height, size.width, size.width * sizeof(PtrElemType));


			cudaMemcpy(m_ptrResultMats_d.data, pMats.data, sizeof(PtrElemType) * size.height * size.width, cudaMemcpyHostToDevice);
			
			delete []  pTmp; // 释放host临时内存区


        }


		size = Wrapper::Instance().GetMatSize("RandonInMat");


		cudaMalloc(&p, TotalBytes(size));


		m_ptrRandonIn_d = make_PtrStepSz<ElemType>(p, size.height, size.width, size.width * sizeof(ElemType));


		size = Wrapper::Instance().GetMatSize("RandonOutMat");


		cudaMalloc(&p, TotalBytes(size));


		m_ptrRandonOut_d = make_PtrStepSz<ElemType>(p, size.height, size.width, size.width * sizeof(ElemType));



    }





    void CudaComputer::FreeCudaMem()
    {

        if (m_ptrStrainMat_d.ptr())
        {
            cudaFree(m_ptrStrainMat_d.data);
        }


        if (m_ptrDispMat_d.ptr())
        {
            cudaFree(m_ptrDispMat_d.data);
        }


        if (m_ptrInFrame_d.ptr())
        {
            cudaFree(m_ptrInFrame_d.data);
        }

		if (m_ptrFilterBuf_d.data)
		{
			cudaFree(m_ptrFilterBuf_d.data);
		}


		if (m_ptrBpParams_d.data)
		{
			cudaFree(m_ptrBpParams_d.data);
		}


		if (m_ptrLpParams_d.data)
		{
			cudaFree(m_ptrLpParams_d.data);
		}


		if (m_ptrMatchParams_d.data)
		{
			cudaFree(m_ptrMatchParams_d.data);
		}


		if (m_ptrRandonIn_d.data)
		{
			cudaFree(m_ptrRandonIn_d.data);
		}


		if (m_ptrRandonOut_d.data)
		{
			cudaFree(m_ptrRandonOut_d.data);
		}


        if (m_ptrResultMats_d.data)
        {

            int c, r;

            cv::Size size     = Wrapper::Instance().GetMatSize("ResultMats");


            PtrElemType  *ptr = new PtrElemType[size.width * size.height];


			cudaMemcpy(ptr, m_ptrResultMats_d.data, sizeof(PtrElemType) * size.height * size.width, cudaMemcpyDeviceToHost);


            for (r = 0; r < size.height; r++)
            {
                for (c = 0; c < size.width; c++)
                {

                    cudaFree(ptr[r * size.width + c]);

                }
            }

			cudaFree(m_ptrResultMats_d.data);


			delete [] ptr;


        }
    }




    void  CudaComputer::DoCalcStrain(const cv::Mat &in, int count, cv::Mat &strainMat)
    {

        assert(in.isContinuous());


        memcpy(m_ptrDispMat.ptr(), in.ptr(), TotalBytes(in));


        cudaMemcpy(m_ptrDispMat_d.ptr(), m_ptrDispMat.ptr(), TotalBytes(m_ptrDispMat), cudaMemcpyHostToDevice);


        dim3 blockDim(k_nThreadDimX, k_nThreadDimY);


        dim3 gridDim(divUp(in.cols, blockDim.x), divUp(in.rows, blockDim.y));


        calcStrainKernal<<<gridDim, blockDim>>>(m_ptrDispMat_d, count, m_ptrStrainMat_d);


        cudaThreadSynchronize();


        cudaMemcpy(m_ptrStrainMat.ptr(), m_ptrStrainMat_d.ptr(), TotalBytes(m_ptrStrainMat), cudaMemcpyDeviceToHost);


		memcpy(strainMat.data, m_ptrStrainMat.data, TotalBytes(m_ptrStrainMat));



    }



    void  CudaComputer::DoCalcDisplacement(const cv::Mat &inData, cv::Mat &dispMat, int win, int step, int scale)
    {
        // copy host data inData -> device memory

		memcpy(m_ptrInFrame.data, inData.data, TotalBytes(inData));

        cudaMemcpy(m_ptrInFrame_d.ptr(), m_ptrInFrame.ptr(), TotalBytes(inData), cudaMemcpyHostToDevice);


        cv::Size size = Wrapper::Instance().GetMatSize("DispMat");


        dim3  threads(k_nThreadDimX, k_nThreadDimY);


        dim3  blocks(divUp(size.width, threads.x), divUp(size.height, threads.y));


        calcDispKernal<<<blocks, threads>>>(m_ptrInFrame_d, m_ptrDispMat_d, m_ptrResultMats_d, win, step, scale);


        cudaThreadSynchronize();

        // copy device memory -> host displacement mat

        cudaMemcpy(m_ptrDispMat.ptr(), m_ptrDispMat_d.ptr(), TotalBytes(dispMat), cudaMemcpyDeviceToHost);


		memcpy(dispMat.data, m_ptrDispMat.data, TotalBytes(dispMat));


    }





    void CudaComputer::DoBpFilter(cv::Mat &in_out)
    {

		DoFilter_1(in_out, m_ptrBpParams_d);

    }



    void CudaComputer::DoLpFilter(cv::Mat &in_out)
    {

		DoFilter_1(in_out, m_ptrLpParams_d);

    }




	//////////////////////////////////////////////////////////////////////////
	// 滤波的具体实现
	//////////////////////////////////////////////////////////////////////////
	void CudaComputer::DoFilter_1(cv::Mat &in_out, PtrStepSzf params)
	{

		PtrStepSzf ptrMat         = make_PtrStepSz<ElemType>(m_ptrInFrame.data, in_out.rows, in_out.cols, sizeof(ElemType) * in_out.cols);

		
		PtrStepSzf ptrMat_d       = make_PtrStepSz<ElemType>(m_ptrInFrame_d.data, in_out.rows, in_out.cols, sizeof(ElemType) * in_out.cols);

		
		PtrStepSzf ptrFilterBuf_d = make_PtrStepSz<ElemType>(m_ptrFilterBuf_d.data, in_out.rows, in_out.cols, sizeof(ElemType) * in_out.cols);




        cudaMemcpy(ptrMat.ptr(), in_out.data, TotalBytes(in_out), cudaMemcpyHostToHost);


        cudaMemcpy(ptrFilterBuf_d.ptr(), ptrMat.ptr(), TotalBytes(in_out), cudaMemcpyHostToDevice);


        dim3  threads(k_nThreadDimX, k_nThreadDimY);


        dim3  blocks(divUp(in_out.cols, threads.x), divUp(in_out.rows, threads.y));


        filterKernal_front<<<blocks, threads>>>(ptrFilterBuf_d, params, ptrMat_d);

        cudaThreadSynchronize();

        cudaMemcpy(ptrFilterBuf_d.ptr(), ptrMat_d.ptr(), TotalBytes(in_out), cudaMemcpyDeviceToDevice);


        filterKernal_back<<<blocks, threads>>>(ptrFilterBuf_d, params, ptrMat_d);

        cudaThreadSynchronize();

        cudaMemcpy(ptrMat.ptr(), ptrMat_d.ptr(), TotalBytes(in_out), cudaMemcpyDeviceToHost);


        cudaMemcpy(in_out.ptr(), ptrMat.ptr(), TotalBytes(in_out), cudaMemcpyHostToHost);


	}





	void CudaComputer::DoRandonSum(const cv::Mat &in, cv::Mat &out)
	{

		//为了加快数据移动的速度，需要使用页锁定内存；
		memcpy(m_ptrInFrame.data, in.data, TotalBytes(in));


		cudaMemcpy(m_ptrRandonIn_d.data, m_ptrInFrame.data, TotalBytes(in), cudaMemcpyHostToDevice);


		PtrStepSzf ptrIn = make_PtrStepSz<ElemType>(m_ptrRandonIn_d.data, in.rows, in.cols, in.step[0]);


        dim3  threads(k_nThreadDimX, k_nThreadDimY);


        dim3  blocks(divUp(out.cols, threads.x), divUp(out.rows, threads.y));

		
		radon_kernal<<<blocks, threads>>>(ptrIn, m_ptrRandonOut_d);


		cudaMemcpy(m_ptrInFrame.data, m_ptrRandonOut_d.data, TotalBytes(out), cudaMemcpyDeviceToHost);


		memcpy(out.data, m_ptrInFrame.data, TotalBytes(out));

	}




	//////////////////////////////////////////////////////////////////////////
	// 均值滤波
	// Mat in 已经把原始mat扩展了step-1列
	//////////////////////////////////////////////////////////////////////////


	void  CudaComputer::MeanFilter(const cv::Mat &in, cv::Mat &out, int step)
	{

		// 首先把原始数据拷贝到数据区m_ptrInFrame，该区域足够大，可以任意使用
		memcpy(m_ptrInFrame.data, in.data, TotalBytes(in));


		// 把数据从host迁移到device上

	cudaMemcpy(m_ptrInFrame_d.data, m_ptrInFrame.data, TotalBytes(in), cudaMemcpyHostToDevice);


		// 创建基于step的指针，由于数据只是占据m_ptrInFrame_d部分区域，需要新建一个指针

		PtrStepSzf ptrIn = make_PtrStepSz<ElemType>(m_ptrInFrame_d.data, in.rows, in.cols, in.step[0]);


		dim3  threads(k_nThreadDimX, k_nThreadDimY);


		dim3  blocks(divUp(out.cols, threads.x), divUp(out.rows, threads.y));


		MeanFilter_Kernal<<<blocks, threads>>>(ptrIn, m_ptrDispMat_d, step);//结果保存到m_ptrDispMat_d


		// 数据从device迁移到host

	cudaMemcpy(m_ptrDispMat.data, m_ptrDispMat_d.data, TotalBytes(out), cudaMemcpyDeviceToHost);

		// 结果拷贝到out中

		memcpy(out.data, m_ptrDispMat.data, TotalBytes(out));



	}
	





	//////////////////////////////////////////////////////////////////////////
	// match filter, 增强位移信号
	//////////////////////////////////////////////////////////////////////////

    void  CudaComputer::MatchFilter(cv::Mat &disp)
	{
		// 首先把原始数据拷贝到数据区m_ptrInFrame，该区域足够大，可以任意使用

		memcpy(m_ptrInFrame.data, disp.data, TotalBytes(disp));

		// 把数据从host迁移到device上

		cudaMemcpy(m_ptrInFrame_d.data, m_ptrInFrame.data, TotalBytes(disp), cudaMemcpyHostToDevice);

		// 创建基于step的指针，它相当于disp的一个克隆

		PtrStepSzf ptrIn = make_PtrStepSz<ElemType>(m_ptrInFrame_d.data, disp.rows, disp.cols, disp.step[0]);


		dim3  threads(k_nThreadDimX, k_nThreadDimY);

		dim3  blocks(divUp(disp.cols, threads.x), divUp(disp.rows, threads.y));

		MatchFilter_Kernal<<<blocks, threads>>>(ptrIn, m_ptrDispMat_d, m_ptrMatchParams_d);//结果保存到m_ptrDispMat_d


		// 数据从device迁移到host

		cudaMemcpy(m_ptrDispMat.data, m_ptrDispMat_d.data, TotalBytes(disp), cudaMemcpyDeviceToHost);

		// 结果拷贝到disp中

		memcpy(disp.data, m_ptrDispMat.data, TotalBytes(disp));


	}



	//位移叠加
    //void  CudaComputer::Rel2AbsDisp(const cv::Mat &clone, cv::Mat &image)


	void  CudaComputer::Rel2AbsDisp(cv::Mat &image)
	{
		// 首先把原始数据拷贝到数据区m_ptrInFrame，该区域足够大，可以任意使用

		memcpy(m_ptrInFrame.data, image.data, TotalBytes(image));

		// 把数据从host迁移到device上

		cudaMemcpy(m_ptrInFrame_d.data, m_ptrInFrame.data, TotalBytes(image), cudaMemcpyHostToDevice);

		// 创建基于step的指针，它相当于image的一个克隆

		PtrStepSzf ptrIn = make_PtrStepSz<ElemType>(m_ptrInFrame_d.data, image.rows, image.cols, image.step[0]);


		dim3  threads(k_nThreadDimX, k_nThreadDimY);

		dim3  blocks(divUp(image.cols, threads.x), divUp(image.rows, threads.y));

		Rel2Abs_Kernal<<<blocks, threads>>>(ptrIn, m_ptrDispMat_d);//结果保存到m_ptrDispMat_d


		// 数据从device迁移到host

		cudaMemcpy(m_ptrDispMat.data, m_ptrDispMat_d.data, TotalBytes(image), cudaMemcpyDeviceToHost);

		// 结果拷贝到image中

		memcpy(image.data, m_ptrDispMat.data, TotalBytes(image));

	}


}