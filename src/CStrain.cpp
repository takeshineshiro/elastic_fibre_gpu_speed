//////////////////////////////////////////////////////////////////////////
//
//

#include "stdafx.h"
#include "Elasto.h"
#include "CElasto.h"
#include "CStrain.h"
#include "CDisplacement.h"
#include "opencv/highgui.h"
#include "opencv/cv.h"
#include <iostream>
#include <vector>
#include "ElastoCudaExt.h"
#include "FileUtility.h"

using namespace std;

void  RadonSum(const cv::Mat &inMat, cv::Mat &radonMat)
{
    int xstart = 0;
    int xend = inMat.rows;
    int t    = inMat.cols;// time extent

    radonMat = cv::Mat::zeros(t - 1, t, inMat.type());

    int tstart = 0;
    int tend = 0;
    int dx = 0;
    float dt = 0.0f;
    float c = 0.0f;
    float *p1;
    const float *p2;
    for (tstart = 0; tstart < t - 1; tstart ++)
    {
        for (tend = tstart + 1; tend < t; tend ++)
        {
            c = (float)(xend - xstart) / (tend - tstart);
            for (dx = xstart; dx < xend; dx ++)
            {
                dt = tstart + (dx - xstart) / c;
                //radonMat.at<float>(tstart, tend) = radonMat.at<float>(tstart, tend) + inMat.at<float>(dx, (int)dt);
                p1 = radonMat.ptr<float>(tstart, tend);
                p2 = inMat.ptr<float>(dx, (int)dt);
                *p1 = *p1 + *p2;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// �����任
// pmatDisplacement,   rows: disp;  cols: time-extent( lines)
//     ��,��ʾһ����, Ҳ����ʱ�� ��
//     ��,��ʾӦ���ֵ
//////////////////////////////////////////////////////////////////////////
void  RadonSum(const CvMat *pmatDisplacement, CvMat **ppmatRodan)
{
	int xstart = 0;
	int xend = pmatDisplacement->rows;
	int t = pmatDisplacement->cols;// time extent

	CvMat *pmatRodan = cvCreateMat(t - 1, t, pmatDisplacement->type);
	cvZero(pmatRodan);

	int tstart = 0;
	int tend = 0;
	int dx = 0;
	float dt = 0.0f;
	float c = 0.0f;

	for (tstart = 0; tstart < t - 1; tstart ++)
	{
		for (tend = tstart + 1; tend < t; tend ++)
		{
			c = (float)(xend - xstart) / (tend - tstart);
			for (dx = xstart; dx < xend; dx ++)
			{
				dt = tstart + (dx - xstart) / c;
				CV_MAT_ELEM(*pmatRodan, float, tstart, tend) = CV_MAT_ELEM(*pmatRodan, float, tstart, tend)
					+ CV_MAT_ELEM(*pmatDisplacement, float, dx, (int)dt);
			}
		}
	}

	*ppmatRodan = pmatRodan;
}



static void PopFirstVector(std::vector<cv::Point2f> &vec)
{
	std::vector<cv::Point2f> swap_vec(vec);
	std::vector<cv::Point2f>::size_type size = swap_vec.size();
	vec.clear();
	if (size > 1)
	{
		int n = size - 1;
		int i;
		for (i = 0; i < n; i++)
		{
			vec.push_back(swap_vec[i + 1]);
		}
	}
}




static void PushBackVector(std::vector<CvPoint2D32f> & vec, CvPoint2D32f &pt)
{
	vec.push_back(pt);
}




static void PopFirstVector(CvPoint2D32f *pVec, int size)
{
	CvPoint2D32f *pvecSwap = new CvPoint2D32f[size];

	ZeroMemory(pvecSwap, sizeof(CvPoint2D32f) * size);

	memcpy(pvecSwap, pVec + 1, sizeof(CvPoint2D32f) * (size - 1));

	memcpy(pVec, pvecSwap, sizeof(CvPoint2D32f) * size);

	delete [] pvecSwap;
}



void  CStrainPlus::CalcStrain(const EInput &input, EOutput &output)
{

    ConfigParam &config_param = ElastoWorker::Instance().GetConfigParam();

    cv::Mat &outDataMat = ElastoWorker::Instance().GetOutmat();


    std::string filename = input.filepath_s;


    //��С���˷���Ӧ�䣻������ϵĵ㣬�������Ϊ��ȣ�������Ϊλ��

    const int points = config_param.fitline_pts;	//������ϵĵ���

    int image_width = outDataMat.cols;

    cv::Mat strImage(cv::Size(outDataMat.rows, image_width - points + 1), CV_32FC1);//������ʾӦ��, �����outDataMat����ת��,���еߵ�.

    cv::Mat strainMat(strImage.cols, strImage.rows, CV_32FC1);  //Ӧ������������strImage����ת�ã���outDataMat��ͬ


    ComputeStrainValueAndImage(points, strainMat, strImage);

    CvMat tosave = strainMat;

    SaveDataFile("strain.dat", &tosave);//���ڱ���Ӧ������


    //�����任&����в�&����ģ��
    {

        int    win_size  = config_param.windowHW;

        double overlap   = (config_param.windowHW - config_param.step) / (float)config_param.windowHW;  // �غ��ʣ�90%

        double sound_velocity = config_param.acousVel; // �����ٶ�

        double sample_frq = config_param.sampleFreqs;

        double prf = config_param.prf;

        int    dep_start = (config_param.sb_x < 0)  ? 0 : config_param.sb_x;

        int    dep_size  =  (config_param.sb_w < 0) ? strainMat.cols : config_param.sb_w;

        int    dep_end   = dep_start + dep_size - 1;

        int    t_start = (config_param.sb_y < 0) ? 0 : config_param.sb_y;

        int    t_size  = (config_param.sb_h < 0) ? strainMat.rows : config_param.sb_h;

        int    t_end   = t_start + t_size - 1;

        //printf("dep_start:%d, dep_end:%d, dep_size:%d; t_start:%d, t_end:%d, t_size:%d\n", dep_start, dep_end, dep_size, t_start, t_end, t_size);
        cv::Mat matStrainTran(strainMat.cols, strainMat.rows, strainMat.type());// ��strainMatת��

        cv::transpose(strainMat, matStrainTran);


        cv::Point  start;

        cv::Point  end;

        cv::Rect   rect;

        rect.x      = t_start;

        rect.y      = dep_start;

        rect.width  = t_size - 1;

        rect.height = dep_size - 1;

        //printf("rect:%d, %d, %d, %d\n", rect.left, rect.right, rect.top, rect.bottom);

#if USE_CPP

        RadonProcess_cpp(start, end, rect, matStrainTran);

#else
        CvPoint sp = start;
        CvPoint ep = end;
        CRect   rc;
        rc.left  = t_start;
        rc.right = t_end;
        rc.top   = dep_start;
        rc.bottom= dep_end;
        RadonProcess_c(sp, ep, rc, matStrainTran);
        start = sp;
        end   = ep;
#endif

        //printf("s_pt:(%d,%d); e_pt(%d,%d)\n", start.x, start.y, end.x, end.y);
        double v = ((end.y - start.y) * win_size * (1 - overlap) * sound_velocity / sample_frq / 2) 
            / ((end.x - start.x) / prf);

        double e = v * v * 3;

        output.v = (float)v;

        output.e = (float)e;


        // ����б��

        cv::Mat imgStrain(strImage.rows, strImage.cols, CV_8UC3);

        cv::cvtColor(strImage, imgStrain, CV_GRAY2BGR);

        cv::line(imgStrain, start, end, CV_RGB(255,0,0), 2, CV_AA, 0);   //����

        cv::imwrite(filename, imgStrain);

    }

}


void  CStrainPlus::CalcStrain_cuda(const EInput &input, EOutput &output)
{

	ConfigParam &config_param = ElastoWorker::Instance().GetConfigParam();

	cv::Mat &outDataMat       = ElastoWorker::Instance().GetOutmat();


	std::string filename      = input.filepath_s;


	//��С���˷���Ӧ�䣻������ϵĵ㣬�������Ϊ��ȣ�������Ϊλ��

	const int points = config_param.fitline_pts;	//������ϵĵ���


	int image_width = outDataMat.cols;

	cv::Mat strImage(cv::Size(outDataMat.rows, image_width - points + 1), CV_32FC1);//������ʾӦ��, �����outDataMat����ת��,���еߵ�.

	cv::Mat strainMat(strImage.cols, strImage.rows, CV_32FC1);  //Ӧ������������strImage����ת�ã���outDataMat��ͬ

	elasto_cuda::Wrapper::Instance().CalcStrain(outDataMat, points, strainMat, strImage);

	//CvMat tosave = strainMat;
	//SaveDataFile("strain.dat", &tosave);//���ڱ���Ӧ�����ݣ���������ʱû�б�Ҫ���档����ʱ���Դ�

	//�����任&����в�&����ģ��
	{

		int    win_size  = config_param.windowHW;

		double overlap   = (config_param.windowHW - config_param.step) / (float)config_param.windowHW;  // �غ��ʣ�90%

		double sound_velocity = config_param.acousVel; // �����ٶ�

		double sample_frq = config_param.sampleFreqs;

		double prf = config_param.prf;


		int    dep_start = (config_param.sb_x < 0)  ? 0 : config_param.sb_x;

		int    dep_size  =  (config_param.sb_w < 0) ? strainMat.cols : config_param.sb_w;

		int    dep_end   = dep_start + dep_size - 1;

		int    t_start = (config_param.sb_y < 0) ? 0 : config_param.sb_y;

		int    t_size  = (config_param.sb_h < 0) ? strainMat.rows : config_param.sb_h;

		int    t_end   = t_start + t_size - 1;
		
		//printf("dep_start:%d, dep_end:%d, dep_size:%d; t_start:%d, t_end:%d, t_size:%d\n", dep_start, dep_end, dep_size, t_start, t_end, t_size);
		cv::Mat matStrainTran(strainMat.cols, strainMat.rows, strainMat.type());// ��strainMatת��

		cv::transpose(strainMat, matStrainTran);


		cv::Point  start;

		cv::Point  end;

		cv::Rect   rect;

		rect.x      = t_start;

		rect.y      = dep_start;

		rect.width  = t_size - 1;

		rect.height = dep_size - 1;

		//printf("rect:%d, %d, %d, %d\n", rect.left, rect.right, rect.top, rect.bottom);

		RadonProcess_cuda(start, end, rect, matStrainTran);

		//printf("s_pt:(%d,%d); e_pt(%d,%d)\n", start.x, start.y, end.x, end.y);
		double v = ((end.y - start.y) * win_size * (1 - overlap) * sound_velocity / sample_frq / 2) 
			       / ((end.x - start.x) / prf);

		double e = v * v * 3;
		output.v = (float)v;
		output.e = (float)e;

		// ����б��
		cv::Mat imgStrain(strImage.rows, strImage.cols, CV_8UC3);
		cv::cvtColor(strImage, imgStrain, CV_GRAY2BGR);
		cv::line(imgStrain, start, end, CV_RGB(255,0,0), 2, CV_AA, 0);   //����
		cv::imwrite(filename, imgStrain);//����ͼƬ�ļ�������ͨ��ͼ����󴫵ݸ�������ʾ�Ĵ���
	}
}

void  CStrainPlus::ComputeStrainValueAndImage(const cv::Mat &in, int count, cv::Mat &strainMat, cv::Mat &strImage)
{
    //float *tmp;  //��ʱ������ָ��Ӧ��ͼ��ĳһ��
    float  coeff_a1; //�洢ֱ��б�ʣ�һ���
    int    deltadepth = 5;   //��λ���
    float  result[4] = {0.0, 0.0, 0.0, 0.0}; //�洢ֱ����Ͻ��
    int    i, j;
    cv::Point2f pt;
    float *p = 0;

#if 0

    {
        std::vector<cv::Point2f> points_vec;

        for(i = 0; i < strImage.cols; i++)// srtImage����, width
        {
            for (j = 0; j < count - 1; j++)	//��ѹ��points - 1����
            {
                //p = in.ptr<float>(i, j);
                pt = cv::Point2f(j * deltadepth, *in.ptr<float>(i, j��);
                points_vec.push_back(pt);
            }

            for(j = 0; j < strImage.rows; ++j)// strImage����, height
            {
                int k = j + count - 1;
                //tmp = static_cast<float*>(static_cast<void*>(strImage->imageData + j * strImage->widthStep + sizeof(float) * i));  //ȡӦ��ͼ���Ӧλ��
                //p = in.ptr<float>(i, k);
                pt = cv::Point2f(k * deltadepth, *in.ptr<float>(i, k));
                points_vec.push_back(pt);  //ѹ�����һ����
                cv::Vec4f line;
                cv::fitLine(cv::Mat(points_vec), line, CV_DIST_L2, 0, 0.01, 0.01);//��С�������
                coeff_a1 = line[1] / line[0];   //���ֱ��б�ʣ���Ϊ���ĵ�Ӧ��
                //strainMat.at<float>(i, j) = coeff_a1;  
                p = strainMat.ptr<float>(i, j);
                *p = coeff_a1;
                //*tmp = 100 * coeff_a1;
                p = strImage.ptr<float>(j, i);
                //strImage.at<float>(j, i) = 100 * coeff_a1;
                *p = 100 *coeff_a1;
                PopFirstVector(points_vec);
            }
            points_vec.clear();
        }
    }
#endif

#if 1
    // ���������CvSeq
    {
        CvPoint2D32f *points = new  CvPoint2D32f[count];
        CvMat ptMat = cvMat(1, count, CV_32FC2, points);
        for(i = 0; i < strImage.cols; i++)// srtImage����, width
        {
            for (j = 0; j < count - 1; j++)	//��ѹ��points - 1����
            {
                pt = cvPoint2D32f(j * deltadepth, *in.ptr<float>(i, j));
                points[j] = pt;
            }

            for(j = 0; j < strImage.rows; j++)// strImage����, height
            {
                int k = j + count - 1;
                pt = cvPoint2D32f(k * deltadepth, *in.ptr<float>(i, k));
                points[count- 1] = pt;  //ѹ�����һ����

                cvFitLine(&ptMat, CV_DIST_L2, 0, 0.01, 0.01, result);//��С�������
                coeff_a1 = result[1] / result[0];   //���ֱ��б�ʣ���Ϊ���ĵ�Ӧ��
                
				*strainMat.ptr<float>(i, j) = coeff_a1;
                *strImage.ptr<float>(j, i) = 100 *coeff_a1;

                PopFirstVector(points, count);
            }
        }
        delete [] points;
    }
#endif

}

void  CStrainPlus::ComputeStrainValueAndImage(int count, cv::Mat &strainMat, cv::Mat &strImage)
{
    cv::Mat &outDataMat = ElastoWorker::Instance().GetOutmat();
    ComputeStrainValueAndImage(outDataMat, count, strainMat, strImage);
}

void  CStrainPlus::RadonProcess_cpp(cv::Point &s, cv::Point &e, const cv::Rect &sub_rc, const cv::Mat &matStrain)
{
    ConfigParam  &config_param = ElastoWorker::Instance().GetConfigParam();
	int  radon_num  = config_param.radon_num;
	int  radon_step = config_param.radon_step;
	cv::Rect rect;
	
	int  intpl_multiple = 1; // ��ֵ��������������任
	std::vector<cv::Point> array_pts;
	std::vector<cv::Rect>  array_rects;

	for (int i = 0; i < radon_num; i++)
	{
		rect.x   = sub_rc.x;
		rect.y    = sub_rc.y + i * radon_step;
		rect.width  = sub_rc.width;
		rect.height = sub_rc.height;

		cv::Mat matSub = matStrain(cv::Rect(rect.x, rect.y, rect.width, rect.height));

		//cv::Mat matMultiple(matSub.rows, matSub.cols * intpl_multiple, matSub.type());
        cv::Mat matMultiple;
		cv::resize(matSub, matMultiple, cv::Size(matSub.cols * intpl_multiple, matSub.rows));
        cv::Mat matRadon;
		::RadonSum(matMultiple, matRadon);

		double  min_val;
		double  max_val;
		cv::Point min_loc;
		cv::Point max_loc;
		cv::minMaxLoc(matRadon, &min_val, &max_val, &min_loc, &max_loc);
		array_pts.push_back(max_loc);
		array_rects.push_back(rect);
	}

	std::sort(array_pts.begin(), array_pts.end(), MyLessThan());

	if (config_param.calc_type.compare("middle") == 0)
	{
		int size = array_pts.size();
		s.x = array_pts[size / 2].y / intpl_multiple;
		s.y = sub_rc.y;

		e.x = array_pts[size / 2].x / intpl_multiple;
		e.y = sub_rc.y + sub_rc.height;
	}
	else if (config_param.calc_type.compare("max") == 0)
	{
		int size = array_pts.size();
		s.x = array_pts[0].y / intpl_multiple;
		s.y = sub_rc.y;

		e.x = array_pts[0].x / intpl_multiple;
		e.y = sub_rc.y + sub_rc.height;
	}
	else if (config_param.calc_type.compare("min") == 0)
	{
		int size = array_pts.size();
		s.x = array_pts[size - 1].y / intpl_multiple;
		s.y = sub_rc.y;

		e.x = array_pts[size - 1].x / intpl_multiple;
		e.y = sub_rc.y + sub_rc.height;
	}
	else
	{
		// middle
	}
}

void  CStrainPlus::RadonProcess_cuda(cv::Point &s, cv::Point &e, const cv::Rect &sub_rc, const cv::Mat &matStrain)
{
    ConfigParam  &config_param = ElastoWorker::Instance().GetConfigParam();
	int  radon_num  = config_param.radon_num;
	int  radon_step = config_param.radon_step;
	cv::Rect rect;
	
	int  intpl_multiple = 1; // ��ֵ��������������任
	std::vector<cv::Point> array_pts;
	std::vector<cv::Rect>  array_rects;

	for (int i = 0; i < radon_num; i++)
	{
		rect.x      = sub_rc.x;
		rect.y      = sub_rc.y + i * radon_step;
		rect.width  = sub_rc.width;
		rect.height = sub_rc.height;

		cv::Mat matSub = matStrain(cv::Rect(rect.x, rect.y, rect.width + 1, rect.height + 1));

        cv::Mat matMultiple;
		cv::resize(matSub, matMultiple, cv::Size(matSub.cols * intpl_multiple, matSub.rows));
        cv::Mat matRadon;
		elasto_cuda::Wrapper::Instance().RadonSum(matMultiple, matRadon);

		double  min_val;
		double  max_val;
		cv::Point min_loc;
		cv::Point max_loc;
		cv::minMaxLoc(matRadon, &min_val, &max_val, &min_loc, &max_loc);
		array_pts.push_back  (max_loc);
		array_rects.push_back(rect);
	}

	std::sort(array_pts.begin(), array_pts.end(), MyLessThan());

	if (config_param.calc_type.compare("middle") == 0)
	{
		int size = array_pts.size();
		s.x = array_pts[size / 2].y / intpl_multiple;
		s.y = sub_rc.y;

		e.x = array_pts[size / 2].x / intpl_multiple;
		e.y = sub_rc.y + sub_rc.height;
	}
	else if (config_param.calc_type.compare("max") == 0)
	{
		int size = array_pts.size();
		s.x = array_pts[0].y / intpl_multiple;
		s.y = sub_rc.y;

		e.x = array_pts[0].x / intpl_multiple;
		e.y = sub_rc.y + sub_rc.height;
	}
	else if (config_param.calc_type.compare("min") == 0)
	{
		int size = array_pts.size();
		s.x = array_pts[size - 1].y / intpl_multiple;
		s.y = sub_rc.y;

		e.x = array_pts[size - 1].x / intpl_multiple;
		e.y = sub_rc.y + sub_rc.height;
	}
	else
	{
		// middle
	}
}

void  CStrainPlus::RadonProcess_c(CvPoint &s, CvPoint &e, const CRect &sub_rc, const CvMat &matStrain)
{
    ConfigParam  &config_param = ElastoWorker::Instance().GetConfigParam();
    int   radon_num  = config_param.radon_num;
    int   radon_step = config_param.radon_step;
	CRect rect;
	
	int  intpl_multiple = 1; // ��ֵ��������������任
	std::vector<CvPoint> array_pts;
	std::vector<CRect>   array_rects;

	for (int i = 0; i < radon_num; i++)
	{
		rect.left   = sub_rc.left;
		rect.top    = sub_rc.top + i * radon_step;
		rect.right  = sub_rc.right;
		rect.bottom = sub_rc.bottom + i * radon_step;

		CvMat *pmatSub = cvCreateMatHeader(rect.Height(), rect.Width(), matStrain.type);
		cvGetSubRect(&matStrain, pmatSub, cvRect(rect.left, rect.top, rect.Width(), rect.Height()));

		CvMat *pmatRadon = 0;
		CvMat *pmatMultiple = cvCreateMat(pmatSub->rows, pmatSub->cols * intpl_multiple, pmatSub->type);
		cvResize(pmatSub, pmatMultiple);
		::RadonSum(pmatMultiple, &pmatRadon);

		double  min_val;
		double  max_val;
		CvPoint min_loc;
		CvPoint max_loc;
		cvMinMaxLoc(pmatRadon, &min_val, &max_val, &min_loc, &max_loc);
		array_pts.push_back(max_loc);
		array_rects.push_back(rect);

		cvReleaseMat(&pmatRadon);
		cvReleaseMat(&pmatMultiple);
		cvReleaseMatHeader(&pmatSub);
	}

	std::sort(array_pts.begin(), array_pts.end(), MyLessThan());

	if (config_param.calc_type.compare("middle") == 0)
	{
		int size = array_pts.size();
		s.x = array_pts[size / 2].y / intpl_multiple;
		s.y = sub_rc.top;

		e.x = array_pts[size / 2].x / intpl_multiple;
		e.y = sub_rc.bottom;
	}
	else if (config_param.calc_type.compare("max") == 0)
	{
		int size = array_pts.size();
		s.x = array_pts[0].y / intpl_multiple;
		s.y = sub_rc.top;

		e.x = array_pts[0].x / intpl_multiple;
		e.y = sub_rc.bottom;
	}
	else if (config_param.calc_type.compare("min") == 0)
	{
		int size = array_pts.size();
		s.x = array_pts[size - 1].y / intpl_multiple;
		s.y = sub_rc.top;

		e.x = array_pts[size - 1].x / intpl_multiple;
		e.y = sub_rc.bottom;
	}
	else
	{
		// middle
	}
}
