#include <stdio.h>
#include <algorithm>
#include <vector>
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <net.h>

#include "mtcnn.h"
#include "det1.id.h"
#include "det1.mem.h"
#include "det2.id.h"
#include "det2.mem.h"
#include "det3.id.h"
#include "det3.mem.h"

using namespace std;
using namespace cv;

float mtcnn::nms_threshold[3] = {0.5, 0.7, 0.7};
float mtcnn::threshold[3] = {0.7, 0.8, 0.9};
float mtcnn::nms_threshold_singleface[3] = {0.5, 0.7, 0.7};
float mtcnn::nms_threshold_mulface[3] = {0.5, 0.7, 0.7};
float mtcnn::nms_threshold_register[3] = {0.5, 0.7, 0.7};

float mtcnn::mean_vals[3] = {127.5, 127.5, 127.5};
float mtcnn::norm_vals[3] = {0.0078125, 0.0078125, 0.0078125};

bool cmpScore(orderScore lsh, orderScore rsh)
{
    if(lsh.score<rsh.score)
        return true;
    else
        return false;
}

mtcnn::mtcnn(int minsize)
{
	Pnet.load_param(det1_param_bin);
    Pnet.load_model(det1_bin);
    Rnet.load_param(det2_param_bin);
    Rnet.load_model(det2_bin);
    Onet.load_param(det3_param_bin);
    Onet.load_model(det3_bin);
	min_size = minsize;
}

void mtcnn::reset_minsize(int minsize)
{
	min_size = minsize;
}

int mtcnn::get_minsize_value(void)
{
	return min_size;
}

void mtcnn::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, std::vector<orderScore>& bboxScore_, float scale)
{
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    float *p = score.channel(1);//score.data + score.cstep;
#ifdef USE_NCNN_NEW_VERSION
    float *plocal = (float *)location.data;
#else
    float *plocal = location.data;
#endif
    Bbox bbox;
    orderScore order;
    for(int row=0;row<score.h;row++)
    {
        for(int col=0;col<score.w;col++)
	{
            if(*p>threshold[0])
	    {
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*col+1)/scale);
                bbox.y1 = round((stride*row+1)/scale);
                bbox.x2 = round((stride*col+1+cellsize)/scale);
                bbox.y2 = round((stride*row+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=location.channel(channel)[0];
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
}

void mtcnn::nms(std::vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname)
{
    if(boundingBox_.empty())
	{
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    while(bboxScore_.size()>0)
   {
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if(order<0)continue;
        heros.push_back(order);
        boundingBox_.at(order).exist = false;//delete it

        for(int num=0;num<boundingBox_.size();num++)
	{
            if(boundingBox_.at(num).exist)
	    {
                //the iou
                maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1)?boundingBox_.at(num).x1:boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1)?boundingBox_.at(num).y1:boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2)?boundingBox_.at(num).x2:boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2)?boundingBox_.at(num).y2:boundingBox_.at(order).y2;
                //maxX1 and maxY1 reuse 
                maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
                maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if(!modelname.compare("Union"))
                    IOU = IOU/(boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if(!modelname.compare("Min"))
				{
                    IOU = IOU/((boundingBox_.at(num).area<boundingBox_.at(order).area)?boundingBox_.at(num).area:boundingBox_.at(order).area);
                }
                if(IOU>overlap_threshold)
		{
                    boundingBox_.at(num).exist=false;
                    for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++)
		    {
                        if((*it).oriOrder == num) 
			{
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }
    for(int i=0;i<heros.size();i++)
        boundingBox_.at(heros.at(i)).exist = true;
}

void mtcnn::refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width)
{
    if(vecBbox.empty())
    {
        cout<<"Bbox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++)
    {
        if((*it).exist)
	{
            bbw = (*it).x2 - (*it).x1 + 1;
            bbh = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[0]*bbw;
            y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
            x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
            y2 = (*it).y2 + (*it).regreCoord[3]*bbh;

            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
          
            maxSide = (h>w)?h:w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if((*it).x1<0)(*it).x1=0;
            if((*it).y1<0)(*it).y1=0;
            if((*it).x2>width)(*it).x2 = width - 1;
            if((*it).y2>height)(*it).y2 = height - 1;

            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}

void mtcnn::cleanUp() 
{
  firstBbox_.clear();
  firstOrderScore_.clear();
  secondBbox_.clear();
  secondBboxScore_.clear();
  thirdBbox_.clear();
  thirdBboxScore_.clear();
}

//优化手段：1.针对注册检测算法，只能检测一个人，检到人退出之后的循环，节省时间
//备注：注册检测算法只针对注册图片中只有一个人。
void mtcnn::mtcnn_oneperson_detect(Mat &image_rgb, vector<cv::Rect> &rect_res, vector<vector<cv::Point2d> > &points)
{
    cleanUp();
    static int fps = 0;
    fps++;
    ncnn::Mat img_;
    //深拷贝一张Mat
    Mat image_tmp;
    image_rgb.copyTo(image_tmp);
    //构造NCNN的Mat
    //img_ = ncnn::Mat::from_pixels(image_tmp.data, ncnn::Mat::PIXEL_BGR2RGB, image_tmp.cols, image_tmp.rows);
    img_ = ncnn::Mat::from_pixels(image_tmp.data, ncnn::Mat::PIXEL_BGR2RGB, image_tmp.cols, image_tmp.rows);
    img = img_;
    img_w = img.w;
    img_h = img.h;
    //减均值除方差，归一化
    img.substract_mean_normalize(mean_vals, norm_vals);
    //计算图像金字塔乘法因子
    float minl = img_w<img_h?img_w:img_h;
    int MIN_DET_SIZE = 12;
    float m = (float)MIN_DET_SIZE/min_size;
    minl *= m;
    float factor = 0.80;
    int factor_count = 0;
    vector<float> scales_;
    vector<float> scales_extend;
    while(minl>MIN_DET_SIZE)
    {
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    orderScore order;
    int count = 0;
    //先跑小图，再跑大图（scales_倒序），这样速度快些
    scales_extend.clear();
    for (int i = 0; i<scales_.size();i++)
    {
	scales_extend.push_back(scales_[scales_.size()-i-1]);
    }
	//循环跑图像金字塔
	for (size_t i = 0; i < scales_extend.size(); i++) 
	{
		//printf("=====>>>>>scales_extend.size=%d,scales_extend[%d]=%f\n",scales_extend.size(),i,scales_extend[i]);
		//第一层：进入Pnet网络
	int hs = (int)ceil(img_h*scales_extend[i]);
	int ws = (int)ceil(img_w*scales_extend[i]);
	ncnn::Mat in;
	resize_bilinear(img_, in, ws, hs);
		//gettimeofday(&tm_before,NULL);
	ncnn::Extractor ex = Pnet.create_extractor();
	ex.set_light_mode(true);
#ifdef USE_TWO_THREAD
	ex.set_num_threads(2);
#else
	ex.set_num_threads(1);
#endif
	ex.input(det1_param_id::LAYER_data, in);
	ncnn::Mat score_, location_;
	ex.extract(det1_param_id::BLOB_prob1, score_);
	ex.extract(det1_param_id::BLOB_conv4_2, location_);
	//gettimeofday(&tm_after,NULL);
	//printf("=====>>>>>Pnet spend time %d us\n",(tm_after.tv_sec-tm_before.tv_sec)*1000000+(tm_after.tv_usec-tm_before.tv_usec));
	std::vector<Bbox> boundingBox_;
	std::vector<orderScore> bboxScore_;

	generateBbox(score_, location_, boundingBox_, bboxScore_, scales_extend[i]);
	nms(boundingBox_, bboxScore_, nms_threshold_register[0]);	

	count = 0;
	for(vector<Bbox>::iterator it=boundingBox_.begin(); it!=boundingBox_.end();it++)
	{
		if((*it).exist)
		{
			//printf("%d jinzita,[%d,%d],[%d,%d]\n",i,(*it).x1,(*it).y1,(*it).x2,(*it).y2);
			firstBbox_.push_back(*it);
			order.score = (*it).score;
			order.oriOrder = count;
			firstOrderScore_.push_back(order);
			count++;
	    }
	}
	bboxScore_.clear();
	boundingBox_.clear();

	//the first stage's nms
	if(count<1)
	{	
		cleanUp();
		continue;
	}
	nms(firstBbox_, firstOrderScore_, nms_threshold_register[0]);	
	refineAndSquareBbox(firstBbox_, img_h, img_w);

	//第二步：进入Rnet网络
	count = 0;
	for(vector<Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++)
	{
		if((*it).exist)
		{
			ncnn::Mat tempIm;
			if((*it).y1 > img_h || (*it).x1 > img_w)
			{
				(*it).exist=false;
				continue;
			}
			//printf("Pnet Out: [%d,%d,%d,%d]\n",(*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);	
			copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
			ncnn::Mat in;
			resize_bilinear(tempIm, in, 24, 24);

			ncnn::Extractor ex = Rnet.create_extractor();
			//gettimeofday(&tm_before,NULL);
			ex.set_light_mode(true);
		#ifdef USE_TWO_THREAD
			ex.set_num_threads(2);
		#else
			ex.set_num_threads(1);
		#endif
			ex.input(det2_param_id::LAYER_data, in);
			ncnn::Mat score, bbox;
			ex.extract(det2_param_id::BLOB_prob1, score);
			ex.extract(det2_param_id::BLOB_conv5_2, bbox);
			//gettimeofday(&tm_after,NULL);
			//printf("=====>>>>>Rnet spend time %d ms\n",(tm_after.tv_sec-tm_before.tv_sec)*1000+(tm_after.tv_usec-tm_before.tv_usec)/1000);
			#ifdef USE_NCNN_NEW_VERSION
			float* ptr = (float *)(score.channel(0));
			if(ptr[1]>threshold[1])
			{
				for(int channel=0;channel<4;channel++)
				{
					//printf("Rnet::regreCoord:%f\n",bbox.channel(0)[channel]);
					it->regreCoord[channel]=bbox.channel(0)[channel];//*(bbox.data+channel*bbox.cstep);
				}
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = ptr[1];//*(score.data+score.cstep);
				//printf("it->score = %f\n",it->score);
				secondBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				secondBboxScore_.push_back(order);
			}
			#else
			if(*(score.data+score.cstep)>threshold[1])
			{
				for(int channel=0;channel<4;channel++)
				{
					it->regreCoord[channel]=bbox.channel(channel)[0];//*(bbox.data+channel*bbox.cstep);
				}
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = score.channel(1)[0];//*(score.data+score.cstep);
				secondBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				secondBboxScore_.push_back(order);
			}
			#endif
			else
			{
				(*it).exist=false;
			}
		}
	}
	firstBbox_.clear();
	firstOrderScore_.clear();

	if(count<1)
	{
		cleanUp();
		continue;
	}

	nms(secondBbox_, secondBboxScore_, nms_threshold_register[1]);
	refineAndSquareBbox(secondBbox_, img_h, img_w);

	//第三步：进入Onet网络 
	count = 0;
	for(vector<Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++)
	{
		if((*it).exist)
		{
			ncnn::Mat tempIm;
			copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
			ncnn::Mat in;
			resize_bilinear(tempIm, in, 48, 48);

			ncnn::Extractor ex = Onet.create_extractor();
			ex.set_light_mode(true);
	#ifdef USE_TWO_THREAD
			ex.set_num_threads(2);
	#else
			ex.set_num_threads(1);
	#endif
			ex.input(det3_param_id::LAYER_data, in);
			ncnn::Mat score, bbox, keyPoint;
			ex.extract(det3_param_id::BLOB_prob1, score);
			ex.extract(det3_param_id::BLOB_conv6_2, bbox);
			ex.extract(det3_param_id::BLOB_conv6_3, keyPoint);

			#ifdef USE_NCNN_NEW_VERSION
			if(score.channel(0)[1]>threshold[2])
			{
				for(int channel=0;channel<4;channel++)
				{
					it->regreCoord[channel]=bbox.channel(0)[channel];
				}
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = score.channel(0)[1];
				for(int num=0;num<5;num++)
				{
					(it->ppoint)[num] = it->x1 + (it->x2 - it->x1)*keyPoint.channel(0)[num];
					(it->ppoint)[num+5] = it->y1 + (it->y2 - it->y1)*keyPoint.channel(0)[num+5];
				}
				thirdBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				thirdBboxScore_.push_back(order);
			}
			#else
			if(score.channel(1)[0]>threshold[2])
			{
				for(int channel=0;channel<4;channel++)
				{
					it->regreCoord[channel]=bbox.channel(channel)[0];
				}
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = score.channel(1)[0];
				for(int num=0;num<5;num++)
				{
					(it->ppoint)[num] = it->x1 + (it->x2 - it->x1)*keyPoint.channel(num)[0];
					(it->ppoint)[num+5] = it->y1 + (it->y2 - it->y1)*keyPoint.channel(num+5)[0];
				}

				thirdBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				thirdBboxScore_.push_back(order);
			}
			#endif
			else
			{
				(*it).exist=false;
			}        
		}
	}
	secondBbox_.clear();
	secondBboxScore_.clear();

	if(count<1)
	{
		cleanUp();
		continue;
	}

	refineAndSquareBbox(thirdBbox_, img_h, img_w);
	nms(thirdBbox_, thirdBboxScore_, nms_threshold_register[2], "Min");

	rect_res.clear();
	//返回最终结果
	for(vector<Bbox>::iterator it=thirdBbox_.begin(); it!=thirdBbox_.end();it++)
	{
		if((*it).exist)
		{			
			Rect face_rect;

			face_rect.x = (*it).x1;
			face_rect.y = (*it).y1;
			face_rect.width = (*it).x2-(*it).x1;
			face_rect.height = (*it).y2-(*it).y1;

			//printf("=====>>>>>[%d,%d,%d,%d]\n",face_rect.x,face_rect.y,face_rect.width,face_rect.height);
			//rectangle(image_tmp, face_rect, Scalar(0,0,255), 2,8,0);
			//char tmp_name[128]={0};
			//sprintf(tmp_name,"result_%d.bmp",fps);

			rect_res.push_back(face_rect);
			vector<cv::Point2d> vp2d;
			for(int num=0;num<5;num++)
			{
				cv::Point2d vpd((*it).ppoint[num],(*it).ppoint[num+5]);
				vp2d.push_back(vpd);
				//cv::circle(image_tmp,Point((*it).ppoint[2*num],(*it).ppoint[2*num+1]),3,Scalar(0,255,255), -1);
			}
			points.push_back(vp2d);
			//imwrite(tmp_name,image_tmp);
		}
	}

	thirdBbox_.clear();
	thirdBboxScore_.clear();
	if(rect_res.size()>0)
	{
		break;
	}
}
}

void mtcnn::clearNet()
{
	Pnet.clear();
	Rnet.clear();
	Onet.clear();
}



















