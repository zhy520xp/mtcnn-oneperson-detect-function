#ifndef MTCNN_H
#define MTCNN_H

#include <vector>
#include <net.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    float ppoint[10];
    float regreCoord[4];
};

struct orderScore
{
    float score;
    int oriOrder;
};

bool cmpScore(orderScore lsh, orderScore rsh);

class mtcnn
{
public:
    mtcnn(int minsize);
	void reset_minsize(int minsize);
	int get_minsize_value(void);
    //void detect(ncnn::Mat& img_, std::vector<face_etem> &ret_res);
	void mtcnn_oneperson_detect(Mat &image, vector<cv::Rect> &rect_res, vector<vector<cv::Point2d> > &points);
	void clearNet();
	~mtcnn(){
		clearNet();			
	};

private:
    void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, vector<orderScore>& bboxScore_, float scale);
    void nms(vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname="Union");
    void refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width);
    void cleanUp();

    ncnn::Net Pnet, Rnet, Onet;
    ncnn::Mat img;

	static float nms_threshold[3];
	static float threshold[3];
	static float nms_threshold_singleface[3];
	static float nms_threshold_mulface[3];
    static float nms_threshold_register[3];


    static float mean_vals[3];
    static float norm_vals[3];

    std::vector<Bbox> firstBbox_, secondBbox_,thirdBbox_;
    std::vector<orderScore> firstOrderScore_, secondBboxScore_, thirdBboxScore_;
    int img_w, img_h;
	int min_size;
};


#endif
