#include <stdio.h>
#include <stdint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <caffe/caffe.hpp>

#include "driveassist.hpp"

#define LANE_USE_KALMAN 1



/// 高速路
/*
#define SRC "file:///media/TOURO/PAPAGO/94750223/16290029.MOV"
int roiX = 195 * 2;
int roiY = 258 * 2;
int roiWidth = 384 * 2;
int roiHeight = 99 * 2;
int srcX1 = 161 * 2;
int KFStateL = 335;
int KFStateR = 450;
*/


/// 城西路

#define SRC "file:///media/TOURO/PAPAGO/94750223/17050041.MOV"
int roiX = 195 * 2;
int roiY = 258 * 2;
int roiWidth = 384 * 2;
int roiHeight = 99 * 2;
int srcX1 = 161 * 2;
int KFStateL = 335;
int KFStateR = 450;
int roadmarkROIX = 345;
int roadmarkROIY = 0;
int roadmarkROIWidth = 128;
int roadmarkROIHeight = 96;




/// 城北路
/*
#define SRC "file:///media/TOURO/PAPAGO/190CRASH/13510002.MOV"
int roiX = 300;
int roiY = 258 * 2;
int roiWidth = 384 * 2;
int roiHeight = 99 * 2;
int srcX1 = 303;
int KFStateL = 315;
int KFStateR = 445;
int roadmarkROIX = 315;
int roadmarkROIY = 0;
int roadmarkROIWidth = 128;
int roadmarkROIHeight = 96;
*/

//#define SRC "file:///media/TOURO/PAPAGO/95450225/17500006.MOV" /** 车辆 */


#define CAR_CASCADE "/media/TOURO/opencv/车辆样本/haarcascade764hog/cascade.xml"
#define ROADMARK_CASCADE "/media/TOURO/opencv/roadmark/data-all-haar/cascade.xml"
#define ROADMARK2_CASCADE "/media/TOURO/opencv/roadmark2/data-all-haar/cascade.xml"

#define NEG_DIR "/media/TOURO/opencv/neg/"


using namespace std;
using namespace cv;

/*
int roiX = 300; //195 * 2;
int roiY = 258 * 2;
int roiWidth = 384 * 2;
int roiHeight = 99 * 2;
int srcX1 = 303; //161 * 2;
*/

/// 手动选择的高斯核
int sigmaX = 4;
int sigmaY = 25;
int gaussianSize = 25;

/*
/// 根据道路标线长度标定的高斯核
int sigmaX = 6;
int sigmaY = 47;
int gaussianSize = 6
*/
int thresholdingQ = 975;
int peakFilterAlpha = 90;
int groupingThreshold = 50;
int ransacIterNum = 50;



/**
 * 车辆探测和测距使用的参数
 */
int mmppx = 100;    /// IPM 图中每个像素的长度，单位：毫米
int carOriginX = 380;    /// imgIPM32 图中汽车前脸的坐标
int carOriginY = 200;    /// imgIPM32 图中汽车前脸的坐标
int carROIX = 0;
int carROIY = 350;
int carROIWidth = 1300;
int carROIHeight = 400;


/**
 * 路面标志探测使用的参数
 */
/*
int roadmarkROIX = 540;
int roadmarkROIY = 508;
int roadmarkROIWidth = 294;
int roadmarkROIHeight = 214;
*/

Mat gaussianKernelX;
Mat gaussianKernelY;

Mat tsfIPM, tsfIPMInv;
Rect roiLane, roiCarDetect, roiRoadmark;

Rect roiRoadmark2 = Rect(384, 1, 118, 118);


const char *winOrigin = "原图";
const char *winROI = "感兴趣区域";
const char *winGray = "灰度图";
const char *winIPM = "俯视变换";
const char *winGaussian = "二维高斯模糊";
const char *winThreshold = "阈值化";
const char *winIPM32 = "俯视变换（彩色）";
const char *winConfig = "参数";
const char *winConfig2 = "参数（2）";
const char *winRoadmark = "路面标线";


Mat frame, imgOrigin, imgROI, imgGray, imgIPM, imgIPM32, imgGaussian, imgThreshold, imgThresholdOld;


#ifdef LANE_USE_KALMAN
LaneKalmanFilter *lkf;
#endif


void onGaussianChange(int _x, void* _ptr) {
    int i;
    float x;
    float xs[gaussianSize] = {0};
    float ys[gaussianSize] = {0};
    float sumx = 0, sumy = 0;
    
    fprintf(stderr, "\\sigam_{x}: %d, \\sigma_{y}: %d, Gaussian Size: %d\n", sigmaX, sigmaY, gaussianSize);
    
    for (i = 0; i < gaussianSize; i++) {
        x = 1.0 * i + 0.5 - gaussianSize * 0.5;
        xs[i] = (1.0 / sigmaX / sigmaX) * exp(-1.0 * x * x / 2 / sigmaX / sigmaX) * (1 - x * x / sigmaX / sigmaX);
        ys[i] = exp(-1.0 * x * x / 2 / sigmaY / sigmaY);
        
        sumx += xs[i];
        sumy += ys[i];
    }
    for (i = 0; i < gaussianSize; i++) {
        xs[i] /= sumx;
        ys[i] /= sumy;
    }
    
    
    gaussianKernelX = Mat(1, gaussianSize, CV_32F, xs).clone();
    gaussianKernelY = Mat(1, gaussianSize, CV_32F, ys).clone();
    
    cout<<"水平高斯核: "<<gaussianKernelX<<"\n";
    cout<<"垂直高斯核: "<<&gaussianKernelY<<"\n";
}

void onKFChange(int _x, void* _ptr) {
    if (lkf == NULL) {
        return;
    }
    
    lkf->setStateLaneL(KFStateL);
    lkf->setStateLaneR(KFStateR);
}

/**
 * 查找一个一维数组中的最大值所在的位置，并按照最大值大小降序排列，返回点的坐标
 * 如果必要，可以进行顶点合并，将 groupingThreshold 个像素邻域的顶点合并
 * 
 * @param Mat&      传入参数，应该是一个一行多列的数组
 * @param Mat&      保存顶点的数组 
 * @param int       最大查找多少个顶点
 */
template <class _T>
void findPeaks(Mat& src, Mat& rst, int num, int groupingThreshold) {
    int i;
    
    src.copyTo(rst);
    rst.empty();
    
    /// 先找出所有局部最大值点
    vector<Point> ps;
    vector<_T> maxs;
    
    for (i = 1; i < src.cols - 1; i++) {
        if (src.at<_T>(0, i - 1) < src.at<_T>(0, i) && src.at<_T>(0, i) > src.at<_T>(0, i + 1)) {
            ps.push_back(Point(i, src.at<_T>(0, i - 1)));
        }
    }
    
    
    /// 如有必要，进行合并
    if (groupingThreshold > 0) {
        /// 取出一个点，判断其与右边点的距离，如果距离小于 groupingThreshold，就合并之，否则继续下一个点
        unsigned int k;
        int grouped;
        
        while (1) {
            grouped = 0;
            
            for (k = 0; k < ps.size() - 1; k++) {
                Point pt1 = ps.at(k);
                Point pt2 = ps.at(k + 1);
                if (pt2.x - pt1.x <= groupingThreshold) {
                    /// 合并两者（删除 k + 1 项，把 k 项替换为合并后的点）
                    Point pt3;
                    
                    pt3.y = (pt1.y + pt2.y) / 2;
                    
                    /// 按照 y 的平方比例计算 x 位置
                    float ratio = pt1.y * pt1.y / (pt1.y * pt1.y + pt2.y * pt2.y + 0.000001);
                    pt3.x = pt1.x * ratio + pt2.x * (1 - ratio);
                    
                    ps.erase(ps.begin() + k);
                    ps.at(k) = pt3;
                    
                    //fprintf(stderr, "Grouping %d & %d\n", pt1.x, pt2.x);
                    
                    grouped = 1;
                    break;  /// 结束本次合并，跳出后重新进行合并
                }
            }
            
            if (grouped == 0) {
                break;
            }
        }
    }
    
    
    /// 依次寻找最大的峰值点，将其推入结果数组
    int cnt = 0;
    while (!ps.empty()) {
        if (cnt >= num) {
            break;
        }
        
        /// 找出当前最大的点，将其推入 dst，然后删除之
        int idx = 0, maxx = 0;
        float curmax = -1;
        for (i = 0; i < (int)ps.size(); i++) {
            if (ps.at(i).y > curmax) {
                idx = i;
                maxx = ps.at(i).x;
                curmax = ps.at(i).y;
            }
        }
        
        maxs.push_back(maxx);
        
        ps.erase(ps.begin() + idx);
    }
    
    
    rst.create(1, maxs.size(), rst.type());
    for (i = 0; i < (int)maxs.size(); i++) {
        rst.at<_T>(0, i) = maxs.at(i);
    }
}

void onROIChange(int _x, void* ptr) {
    /// IPM 仿射变换矩阵
    roiLane = Rect(roiX, roiY, roiWidth, roiHeight);
    
    Point2f src[4];
    Point2f dst[4];
    
    src[0].x = 1;
    src[0].y = 1;
    src[1].x = srcX1;
    src[1].y = roiHeight;
    src[2].x = roiWidth - srcX1;
    src[2].y = roiHeight;
    src[3].x = roiWidth;
    src[3].y = 1;
    
    dst[0].x = 1;
    dst[0].y = 1;
    dst[1].x = 1;
    dst[1].y = roiHeight;
    dst[2].x = roiWidth;
    dst[2].y = roiHeight;
    dst[3].x = roiWidth;
    dst[3].y = 1;
    
    tsfIPM = getPerspectiveTransform(dst, src);
    tsfIPMInv = tsfIPM.inv();
    
    
    /// 汽车检测 ROI
    roiCarDetect.x = carROIX;
    roiCarDetect.y = carROIY;
    roiCarDetect.width = carROIWidth;
    roiCarDetect.height = carROIHeight;
    
    /// 路面标志探测 ROI
    roiRoadmark.x = roadmarkROIX;
    roiRoadmark.y = roadmarkROIY;
    roiRoadmark.width = roadmarkROIWidth;
    roiRoadmark.height = roadmarkROIHeight;

}

/**
 * 传入原图，以及 ROI，会在指定目录下生成一系列 ROI 区域的图片
 */
void cutRegion(Mat *imgInput, Rect _roi, const char* dir) {
    static int i = 0;
    char path[1024] = {0};
    
    
    if (_roi.x < 0 || _roi.x >= imgInput->cols) {
        _roi.x = 0;
    }
    if (_roi.y < 0 || _roi.y >= imgInput->rows) {
        _roi.y = 0;
    }
    if (_roi.x + _roi.width >= imgInput->cols) {
        _roi.width = imgInput->cols - 1 - _roi.x;
    }
    if (_roi.y + _roi.height >= imgInput->rows) {
        _roi.height = imgInput->rows - 1 - _roi.y;
    }
    
    /// 绘制截图区域
    rectangle(imgOrigin, Point(_roi.x, _roi.y), Point(_roi.x + _roi.width, _roi.y + _roi.height), CV_RGB(255, 0, 0), 2);


    snprintf(path, sizeof(path) - 1, "%s/%06d.png", dir, ++i);
    imwrite(path, Mat(*imgInput, _roi));
}


/**
 * 对车道线进行粒子滤波
 * 
 * @return vector<int>    两个坐标，是滤波后的车道线的 x 坐标
 */
template <typename T>
vector<int> lanePF(Mat& _iInput) {
    int i;
    
    static int stateNum = 2;
    static int measureNum = 2;
    static int sampleNum = 2000;
    
    
    static CvMat *lowerBound = cvCreateMat(stateNum, 1, CV_32F);
    static CvMat *upperBound = cvCreateMat(stateNum, 1, CV_32F);
    
    int peakLen = 8;
    int borderLen = 3;
    int totalLen = peakLen + borderLen * 2;
    float stdLine[totalLen] = {0};
    
    Mat iInput(_iInput, Rect(0, 0, _iInput.cols / 2, _iInput.rows));
    
    
    static CvConDensation *con = NULL;
    if (con == NULL) {
        /// 初始化粒子滤波器 
        con = cvCreateConDensation(stateNum, measureNum, sampleNum);
        
        cvmSet(lowerBound, 0, 0, 0.0);
        cvmSet(lowerBound, 1, 0, 0.0);
        
        cvmSet(upperBound, 0, 0, iInput.cols - totalLen - 1);
        cvmSet(upperBound, 1, 0, iInput.rows - totalLen - 1);
        float A[2][2] = {
            1, 0,
            0, 1
        };
        memcpy(con->DynamMatr, A, sizeof(A));
        
        
        cvConDensInitSampleSet(con, lowerBound, upperBound);
        
        CvRNG rng_state = cvRNG(time(NULL));
        for(int i=0; i < sampleNum; i++){
            con->flSamples[i][0] = float(cvRandInt( &rng_state ) % iInput.cols); //width
            con->flSamples[i][1] = float(cvRandInt( &rng_state ) % iInput.rows);//height
        }
        
    }
    
    /// 初始化标准曲线
    for (int i = borderLen - 1; i < borderLen - 1 + peakLen; i++) {
        stdLine[i] = 255;
    }
    
    
    
    /// 计算概率（阈值车道响应拟合法）
    /*
    float maxp = 0;
    for (i = 0; i < sampleNum; i++) {
        int x, y, xp, yp, k;
        float e, p;
        
        x = con->flSamples[i][0];
        y = con->flSamples[i][1];
        
        
        e = 0;
        
        for (k = 0; k < totalLen; k++) {
            xp = x + k;
            yp = y;
            
            float d;
            //printf("k = %d, point=%d, stdlint=%f\n", k, iInput->ptr<T>(yp)[xp], stdLine[k]);
            
            d = iInput.ptr<T>(yp)[xp];
            d = abs(d - stdLine[k]);
            d /= 255.0;
            e += d; 
        }
        e /= 255.0;
        
        p = exp(-1 * e);
        
        if (p > maxp) {
            maxp = p;
        }
        
        con->flConfidence[i] = p;
    }
    */
    
    
    /// 计算概率（垂直像素最多法）
    
    for (i = 0; i < sampleNum; i++) {
        int x, y;
        float e, p;
        
        p = 0; e = 0;
        
        for (y = 0; y < iInput.rows; y++) {
            x = con->flSamples[i][0] + (con->flSamples[i][1] - con->flSamples[i][0]) * (1.0 * y / iInput.rows);
            e += iInput.ptr<T>(y)[x];
        }
        e /= 255.0;
        if (e > iInput.rows) {
            e = 0;
        }
        
        p = exp(-0.1 * (iInput.rows - e));
        con->flConfidence[i] = p;
        //cout<<"x1="<<con->flSamples[i][0]<<", x2="<<con->flSamples[i][1]<<", P="<<p<<endl;
    }
    cout<<"更新前："<<con->State[0]<<endl;
    cvConDensUpdateByTime(con);
    
    
    
    /*
    for (i = 0; i < sampleNum; i++) {
        Point p1(con->flSamples[i][0], con->flSamples[i][1]);
        Point p2(p1.x + totalLen, p1.y);
        
        line(iInput, p1, p2, CV_RGB(128, 128, 128), 1);
    }
    */
    
    line(_iInput, Point(con->State[0], 0), Point(con->State[1], iInput.rows), CV_RGB(128, 128, 128), 1);
    cout<<"预测位置："<<con->State[0]<<endl;


    vector<int> ret;
    
    ret.push_back(con->State[0]);
    ret.push_back(con->State[1]);
    
    return ret;
}




void detectLane(Mat imgInput) {
    // 在原图上绘制 ROI 方框
    rectangle(imgOrigin, Point(roiLane.x, roiLane.y), Point(roiLane.x + roiLane.width, roiLane.y + roiLane.height), CV_RGB(0, 255, 0));    
    
    /// ROI 图
    imgROI = Mat(imgOrigin, roiLane);
    
    
    /// 灰度图
    cvtColor(imgROI, imgGray, CV_RGB2GRAY);
    imshow(winGray, imgGray);
    
    
    /// IPM 图
    warpPerspective(imgGray, imgIPM, tsfIPM, imgROI.size());
    imshow(winIPM, imgIPM);
    
    warpPerspective(imgROI, imgIPM32, tsfIPM, imgROI.size());
    
    cutRegion(&imgIPM32, Rect(0, 0, imgIPM32.cols, imgIPM32.rows), "/media/TOURO/raw");
    return;
    
    
    // 在 ROI 图上绘制 srcX1 和 src 坐标
    line(imgROI, Point(srcX1, 1), Point(srcX1, roiHeight), CV_RGB(0, 255, 0));
    line(imgROI, Point(roiWidth - srcX1, 1), Point(roiWidth - srcX1, roiHeight), CV_RGB(0, 255, 0));
    
    //imshow(winROI, imgROI2);


    /// 高斯模糊
    
    sepFilter2D(imgIPM, imgGaussian, imgIPM.depth(), gaussianKernelX, gaussianKernelY);
    /*
    if (gaussianSize % 2 == 0) {
        gaussianSize += 1;
    }
    GaussianBlur(imgIPM, imgGaussian, Size(gaussianSize, gaussianSize), sigmaX, sigmaY);
    */
    imshow(winGaussian, imgGaussian);
    
    
    
    
    /// 阈值过滤
    Mat imgHist;
    //Mat imgThresholdTmp;
    
    equalizeHist(imgGaussian, imgHist);
    threshold(imgHist, imgThreshold, 255 * thresholdingQ / 1000, 255, THRESH_TOZERO);
    //adaptiveThreshold(imgHist, imgThreshold, 255 * thresholdingQ / 1000, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 5, 4);
    //Canny(imgHist, imgThreshold, 200, 10);

    //imshow(winThreshold, imgThreshold);
    
    //imshow("阈值化（前）", imgThresholdTmp);
    
    //laneFilter(imgThresholdTmp, imgThreshold, 50);
    
    vector<int> pfXs;
    pfXs = lanePF<uint8_t>(imgThreshold);
    
    
    
    /// 计算道路响应曲线
    Mat tsmResp(1, imgThreshold.rows, CV_32FC1, Scalar::all(1));
    Mat t1, resp, oldResp;
    
    imgThreshold.convertTo(t1, CV_32FC1);
    resp = tsmResp * t1;
    
    resp /= 255.0;    /// 计算点的个数
    
    GaussianBlur(Mat(resp), resp, Size(21, 21), 8);
    
    float alpha = peakFilterAlpha / 100.0;  /// alpha 越大则越容易忘记
    resp = resp * alpha + oldResp * (1 - alpha);
    oldResp = resp;
    
    // 绘制响应曲线
    int i;
    Point lp(0, imgGaussian.rows - resp.at<float>(0, 0));
    Point cp;
    for (i = 1; i < resp.cols; i++) {
        cp.x = i;
        cp.y = imgGaussian.rows - resp.at<float>(0, i);
        line(imgGaussian, lp, cp, CV_RGB(255, 255, 255));
        lp = cp;
    }
    
    
    // 计算曲线顶点并绘制
    Mat datPeaks;
    Mat imgThreshold2;
    LineFit lineFit(&imgThreshold, &resp);;
    
    imgThreshold.copyTo(imgThreshold2);
    
    findPeaks<float>(resp, datPeaks, 10, groupingThreshold);
    
    #ifdef LANE_USE_KALMAN
    if (lkf == NULL) {
        lkf = new LaneKalmanFilter(imgIPM32.size());
        onKFChange(0, NULL);
        lkf->setStateLaneL(KFStateL);
        lkf->setStateLaneR(KFStateR);
    }
    
    lkf->next();    /// 开始新一轮迭代
    #endif
    
    
    for (i = 0; i < datPeaks.cols; i++) {
        float x;
        x = datPeaks.at<float>(0, i);
        
        /// 删除图片左边三分之一的区域
        if (x < imgGaussian.cols / 3) {
            continue;
        }
        
        /// 删除 ROI 中心区域附近的匹配
        if (abs(x - imgROI.cols / 2) < 20) {
            continue;
        }
        
        line(imgGaussian, Point(x, 0), Point(x, imgGaussian.rows), CV_RGB(255, 255, 255));
        //line(imgThreshold, Point(x, 0), Point(x, imgGaussian.rows), CV_RGB(255, 255, 255));
        //imshow(winThreshold, imgThreshold);
        
        /// RANSAC 三次贝塞尔曲线拟合道路
        
        int u, v;
        RANSACFit ransacFit;
        for (u = x - 50; u < x + 50; u++) {
            for (v = 0; v < imgThreshold.rows; v++) {
                if (imgGaussian.at<uint8_t>(u, v) > 0) {
                    ransacFit.addPoint(Point(u, v), imgThreshold.ptr<uint8_t>(v)[u]);
                }
            }
        }
        
        
        /// 进行 RANSAC 拟合，然后绘制拟合后的曲线
        /*
        vector<Point3f> psSpline = ransacFit.fit(ransacIterNum, imgThreshold);
        float t;
        Point p, pOld;
        if( psSpline.size() > 0) {
            pOld = ransacFit.getPoint(0, psSpline);
            
            for (t = 0.01; t <= 1; t += 0.01) {
                Point p = ransacFit.getPoint(t, psSpline);
                
                line(imgIPM32, pOld, p, CV_RGB(255, 0, 0));
                pOld = p;
            }
        }
        else {
            cout<<"没有拟合线条"<<endl;
        }
        */
        /// RANSAC 三次贝塞尔曲线拟合结束
        
        
        /// 简单直线拟合道路
        /**
        std::vector<cv::Point2f> ps;
        Vec4f psOut;
        srand(i);
        int r, g, b;
        r = rand() % 255; g = rand() % 255; b = rand() % 255;
        for (u = x - 50; u < x + 50; u++) {
            for (v = 0; v < imgThreshold.rows; v++) {
                if (imgThreshold.ptr<uint8_t>(v)[u] > 0) {
                    line(imgThreshold2, Point(u, v), Point(u, v), CV_RGB(r, g, b));
                    line(imgIPM32, Point(u, v), Point(u, v), CV_RGB(r, g, b));
                    ps.push_back(cv::Point2f(u, v));
                }
            }
        }
        
        if (ps.size() > 0) {
            fitLine(ps, psOut, CV_DIST_L2, 0, 0.01, 0.01);
            float x1, x2, y1, y2;
            x1 = psOut[2];
            y1 = psOut[3];
            x2 = psOut[2] + 300 * psOut[0];
            y2 = psOut[3] + 300 * psOut[1];
            line(imgIPM32, Point(x1, y1), Point(x2, y2), CV_RGB(r, g, b), 2);
            line(imgThreshold2, Point(x1, y1), Point(x2, y2), CV_RGB(r, g, b), 2);
        }
        */
        /// 直线拟合道路结束
        
        
        /// 高级直线拟合道路开始
        
        vector<int> rangeX;
        vector<Point> fitP;
        
        rangeX = lineFit.getRange(x);
        //fprintf(stderr, "Range = (%d, %d) = %d\n", rangeX.at(0), rangeX.at(1), rangeX.at(1) - rangeX.at(0));
        fitP = lineFit.fitLine(rangeX.at(0), rangeX.at(1));
        
        /// 拟合范围（黄线）
        line(imgIPM32, Point(rangeX.at(0), 0), Point(rangeX.at(0), imgIPM32.cols), CV_RGB(255, 255, 0));
        line(imgIPM32, Point(rangeX.at(1), 0), Point(rangeX.at(1), imgIPM32.cols), CV_RGB(255, 255, 0));
        line(imgIPM32, fitP[0], fitP[1], CV_RGB(255, 0, 0));
        
        #ifdef LANE_USE_KALMAN
        lkf->addLine(fitP[0], fitP[1]);
        #endif
        
        /// 在 imgROI 图上画出拟合道路
        vector<Point2f> ps;
        vector<Point2f> psOut;

        ps.push_back(Point2f(fitP[0].x, fitP[0].y));
        ps.push_back(Point2f(fitP[1].x, fitP[1].y));
        
        perspectiveTransform(ps, psOut, tsfIPMInv);
        
        
        line(imgROI, psOut.at(0), psOut.at(1), CV_RGB(255, 0, 255), 2);
        
        /// 在原图上画出拟合道路
        psOut.at(0).x += roiLane.x;
        psOut.at(0).y += roiLane.y;
        psOut.at(1).x += roiLane.x;
        psOut.at(1).y += roiLane.y;
        
        line(imgOrigin, psOut.at(0), psOut.at(1), CV_RGB(255, 0, 255), 2);
        
        /// 高级直线拟合道路结束
        
        
        /// 二次贝塞尔曲线道路拟合开始
        /**
        vector<int> rangeX;
        vector<Point> fitP;
        Point pS, pE, pC;
        float t;
        
        rangeX = lineFit.getRange(x);
        //fprintf(stderr, "Range = (%d, %d) = %d\n", rangeX.at(0), rangeX.at(1), rangeX.at(1) - rangeX.at(0));
        fitP = lineFit.fitBeizer2R(rangeX.at(0), rangeX.at(1));
        pS = fitP[0];
        pE = fitP[1];
        pC = fitP[2];
        
        Point pOld(-1, -1);
        for (t = 0; t <= 1; t += 1.0 / imgIPM32.cols) {
            u = (1 - t) * (1 - t) * pS.x + 2 * t * (1 - t) * pC.x + t * t * pE.x;
            v = (1 - t) * (1 - t) * pS.y + 2 * t * (1 - t) * pC.y + t * t * pE.y;
            
            Point p(u, v);
            if (pOld.x != -1 || pOld.y != -1) {
                line(imgIPM32, pOld, p, CV_RGB(255, 0, 0));
            }
            pOld = p;
        }            
        
        line(imgIPM32, Point(rangeX.at(0), 0), Point(rangeX.at(0), imgIPM32.cols), CV_RGB(255, 255, 0));
        line(imgIPM32, Point(rangeX.at(1), 0), Point(rangeX.at(1), imgIPM32.cols), CV_RGB(255, 255, 0));
        
        
        /// 在 imgROI 图上画出拟合道路
        
        vector<Point2f> ps;
        vector<Point2f> psOut;

        ps.push_back(Point2f(fitP[0].x, fitP[0].y));
        ps.push_back(Point2f(fitP[1].x, fitP[1].y));
        ps.push_back(Point2f(fitP[2].x, fitP[2].y));
        
        perspectiveTransform(ps, psOut, tsfIPMInv);
        
        
        line(imgROI, psOut.at(0), psOut.at(1), CV_RGB(0, 0, 255), 2);
        
        /// 在原图上画出拟合道路
        psOut.at(0).x += roiLane.x;
        psOut.at(0).y += roiLane.y;
        psOut.at(1).x += roiLane.x;
        psOut.at(1).y += roiLane.y;
        psOut.at(2).x += roiLane.x;
        psOut.at(2).y += roiLane.y;
        
        pS = psOut.at(0);
        pE = psOut.at(1);
        pC = psOut.at(2);
        
        pOld.x = pOld.y = -1;
        for (t = 0; t <= 1; t += 1.0 / imgIPM32.cols) {
            u = (1 - t) * (1 - t) * pS.x + 2 * t * (1 - t) * pC.x + t * t * pE.x;
            v = (1 - t) * (1 - t) * pS.y + 2 * t * (1 - t) * pC.y + t * t * pE.y;
            
            Point p(u, v);
            if (pOld.x != -1 || pOld.y != -1) {
                line(imgInput, pOld, p, CV_RGB(255, 0, 255), 2);
            }
            pOld = p;
        }     
        */
        
        /// 二次曲线道路拟合结束

    }
    
    
    #ifdef LANE_USE_KALMAN
    /// 道路标线卡尔曼滤波
    lkf->predict();
    
    
    /// 在 IPM 32 上画出滤波后的曲线
    vector<Point> lLane, rLane, lSLane, rSLane;
    
    lLane = lkf->getPredictL();
    rLane = lkf->getPredictR();
    line(imgIPM32, lLane[0], lLane[1], CV_RGB(128, 64, 255));
    line(imgIPM32, rLane[0], rLane[1], CV_RGB(128, 64, 255));
    
    lSLane = lkf->getStateL();
    rSLane = lkf->getStateR();
    line(imgIPM32, lSLane[0], lSLane[1], CV_RGB(0, 0, 255));
    line(imgIPM32, rSLane[0], rSLane[1], CV_RGB(0, 0, 255));
    
    
    /// 在 imgOrigin 上画出滤波后的车道线和状态车道线
    vector<Point2f> ps;
    vector<Point2f> psOut;
    
    ps.push_back(lLane[0]);
    ps.push_back(lLane[1]);
    ps.push_back(rLane[0]);
    ps.push_back(rLane[1]);
    ps.push_back(lSLane[0]);
    ps.push_back(lSLane[1]);
    ps.push_back(rSLane[0]);
    ps.push_back(rSLane[1]);
    
    perspectiveTransform(ps, psOut, tsfIPMInv);
    
    for (size_t i = 0; i < psOut.size(); i++) {
        psOut.at(i).x += roiLane.x;
        psOut.at(i).y += roiLane.y;
    }

    /// 滤波后车道线
    line(imgOrigin, psOut[0], psOut[1], CV_RGB(128, 128, 255), 4, CV_AA);
    line(imgOrigin, psOut[2], psOut[3], CV_RGB(128, 128, 255), 4, CV_AA);
    /// 目标车道线
    line(imgOrigin, psOut[4], psOut[5], CV_RGB(0, 0, 255), 1, CV_AA);
    line(imgOrigin, psOut[6], psOut[7], CV_RGB(0, 0, 255), 1, CV_AA);

    #endif
    

    /// 在原图上画出粒子滤波车道线
    ps.clear();
    ps.push_back(Point(pfXs[0], 0));
    ps.push_back(Point(pfXs[1], imgThreshold.rows));
    perspectiveTransform(ps, psOut, tsfIPMInv);
    
    for (size_t i = 0; i < psOut.size(); i++) {
        psOut.at(i).x += roiLane.x;
        psOut.at(i).y += roiLane.y;
    }

    /// 粒子车道线
    line(imgOrigin, psOut[0], psOut[1], CV_RGB(255, 0, 0), 2, CV_AA);




    imshow(winGaussian, imgGaussian);
    imshow(winThreshold, imgThreshold2);
    
}



/**
 * 传入原图的灰度图，以及 ROI
 */
void detectCar(Mat *imgInput, Rect _roi) {
    vector<Rect> cars;
    vector<Point2f> pIn, pOut;
    char cTxt[1024] = {0};
    static CascadeClassifier _cascade;
    static CascadeClassifier* cascade = NULL;
    int ret;
    
    if (cascade == NULL) {
        cascade = &_cascade;
        ret = cascade->load(CAR_CASCADE);
        if (!ret) {
            fprintf(stderr, "Could not load cascade file: `%s'\n", CAR_CASCADE);
            exit(-1);
        }
    }
    
    if (_roi.x < 0 || _roi.x >= imgInput->cols) {
        _roi.x = 0;
    }
    if (_roi.y < 0 || _roi.y >= imgInput->rows) {
        _roi.y = 0;
    }
    if (_roi.x + _roi.width >= imgInput->cols) {
        _roi.width = imgInput->cols - 1 - _roi.x;
    }
    if (_roi.y + _roi.height >= imgInput->rows) {
        _roi.height = imgInput->rows - 1 - _roi.y;
    }

    if (carOriginX > imgOrigin.cols) {
        carOriginX = imgOrigin.cols - 1;
    }
    if (carOriginY > imgOrigin.rows) {
        carOriginY = imgOrigin.rows - 1;
    }

    Mat iROI(*imgInput, _roi);
    Mat iGray;
    
    cvtColor(iROI, iGray, COLOR_BGR2GRAY);
    
    
    
    /// 绘制车辆探测区域
    rectangle(imgOrigin, Point(_roi.x, _roi.y), Point(_roi.x + _roi.width, _roi.y + _roi.height), CV_RGB(0, 255, 255), 1);
    
    /// 绘制车辆前脸位置（用一个红色原点在 imgIPM32 上表示）
    circle(imgIPM32, Point(carOriginX, carOriginY), 6, CV_RGB(255, 0, 0), 3, CV_AA);
    
    /// 探测车辆
    cascade->detectMultiScale(iGray, cars, 1.1, 1, 0 | CASCADE_SCALE_IMAGE, Size(30, 30) );
    

    for( size_t i = 0; i < cars.size(); i++ )
    {
        /// 绘制汽车识别框
        Point2f p1, p2, pc, pcT, pcR, pTxt;
        p1.x = cars[i].x + _roi.x;
        p1.y = cars[i].y + _roi.y;
        p2.x = p1.x + cars[i].width;
        p2.y = p1.y + cars[i].height;
        rectangle(imgOrigin, p1, p2, CV_RGB(255, 200, 255), 2);
        
        pTxt.x = (p1.x + p2.x) / 2;
        pTxt.y = (p1.y + p2.y) / 2;
        
        /// 计算汽车距离，先转换到 IPM32 坐标系，然后计算距离
        /// pc 取车辆区域（矩形）的底边的中点
        
        pc.x = (p1.x + p2.x) / 2;
        pc.y = MAX(p1.y, p2.y);
        
        pcR.x = pc.x - roiLane.x;
        pcR.y = pc.y - roiLane.y;
        
        pIn.clear();
        pIn.push_back(pcR);
        perspectiveTransform(pIn, pOut, tsfIPM);
        pcT = pOut[0];
        

        
        float dispx = sqrt((pcT.x - carOriginX) * (pcT.x - carOriginX) + (pcT.y - carOriginY) * (pcT.y - carOriginY));
        float dis = dispx * mmppx / 1000;
        
        
        fprintf(stderr, "Car #%lu, O(%.1f, %.1f), R(%.1f, %.1f), I(%.1f, %.1f), (I)dis: %0.2fpx, %0.2fM\n", i, pc.x, pc.y, pcR.x, pcR.y, pcT.x, pcT.y, dispx, dis);
        
        circle(imgIPM32, pcT, 4, CV_RGB(0, 0, 255), 2);
        circle(imgOrigin, pc, 4, CV_RGB(0, 0, 255), 2);
        
        /// 如果所在点超出 ROI 区域，则不显示距离
        if (!(pc.x >= roiLane.x && pc.x <= roiLane.x + roiLane.width)) {
            continue;
        }
        if (!(pc.y >= roiLane.y && pc.y <= roiLane.y + roiLane.height)) {
            continue;
        }
        
        
        snprintf(cTxt, sizeof(cTxt) - 1, "%0.1fM", dis);
        putText(imgOrigin, String(cTxt), pTxt, FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0, 0, 255));
        snprintf(cTxt, sizeof(cTxt) - 1, "%0.1f", cars[i].width * 1.0 / tan(cars[i].y));
        putText(imgOrigin, String(cTxt), p1, FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 128, 0));
        
        
    }
}


/**
 * 使用 IPM 进行标志检测（卷积神经网络）
 * Detect roadmark on IPM with Convolutional Neural Netrowks
 */
void detectRoadmarkCNN(Mat *imgInput) {
    const char *winDR2IPM = "Detect Roadmark CNN IPM";
    const char *trained_file = "/media/TOURO/caffe/roadmark-test/mynet1_iter_35000.caffemodel";
    const char *model_file = "/media/TOURO/caffe/roadmark-test/mynet1_mem.prototxt";
    
    Mat iROI, iGray, iIPM, iHist, iThres, iGauss;
    
    iROI = Mat(*imgInput, roiLane);
    
    Rect roi = roiRoadmark;
    
    cout<<roi<<endl;
    

    /// IPM 图
    warpPerspective(iROI, iIPM, tsfIPM, iROI.size());
    
    rectangle(iIPM, Point(roi.x, roi.y), Point(roi.x + roi.width, roi.y + roi.height), CV_RGB(128, 128, 255));
    
    
    roi.x = max(roi.x, 0);
    roi.y = max(roi.y, 0);
    if (roi.x + roi.width >= iIPM.cols - 1) {
        fprintf(stderr, "Roadmark 区域 x 轴方向超出范围\n");
        imshow(winDR2IPM, iIPM);
        return;
    }
    if (roi.y + roi.height >= iIPM.rows - 1) {
        fprintf(stderr, "Roadmark 区域 y 轴方向超出范围\n");
        imshow(winDR2IPM, iIPM);
        return;
    }
    
    /// 将图片截取出来，用作训练
    /// Cut ROI for CNN training
    //cutRegion(&iIPM, roi, "/media/TOURO/neg");
    
    
    /// 使用 CNN 进行识别
    Net<float> _cnn(model_file, caffe::TEST);
    Net<float> *cnn = NULL;
    
    if (cnn == NULL) {
        fprintf(stderr, "初始化卷积神经网络\n");
        
        cnn = &_cnn;
        cnn->CopyTrainedLayersFrom(trained_file);
    }
    
    shared_ptr<MemoryDataLayer<float> > md_layer = boost::dynamic_pointer_cast <MemoryDataLayer<float> >(cnn->layers()[0]);
    if (!md_layer) {
        fprintf(stderr, "(Caffe)卷积神经网络的第一层不是内存数据层\n");
        exit(-1);
    }
    
    Mat iCNN;
    resize(Mat(iIPM, roi), iCNN, cv::Size(64, 48));
    
    vector<Mat> images(1, iCNN);
    vector<int> labels(1, 0);
    float loss;
    
    
    md_layer->AddMatVector(images, labels);
    cnn->ForwardPrefilled(&loss);
    
    shared_ptr<Blob<float> > prob = cnn->blob_by_name("prob");
    float maxval = 0;
    int maxidx = 0;
    for (int i = 0; i < prob->count(); i++) {
        float val = prob->cpu_data()[i];
        if (val > maxval) {
            maxval = val;
            maxidx = i;
        }
    }
    
    char txt[1024] = {0};
    snprintf(txt, sizeof(txt) - 1, "Label: %d, Val: %0.4f", maxidx, maxval);
    putText(iIPM, String(txt), Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0, 0, 255));
    

    
    fprintf(stderr, "路标探测：探测到的最大标签是%d, 最大值是 %0.4f\n", maxidx, maxval);
    
    if (maxidx > 0) {
        rectangle(iIPM, Point(roi.x, roi.y), Point(roi.x + roi.width, roi.y + roi.height), CV_RGB(255, 255, 0));
        
        cutRegion(&iIPM, roi, "/media/TOURO/neg/1");
        
        
        /// 在 imgOrigin 上绘制探测到的路标界限
        vector<Point2f> ps, psIn;
        psIn.push_back(Point(roi.x, roi.y));
        psIn.push_back(Point(roi.x + roi.width, roi.y));
        psIn.push_back(Point(roi.x + roi.width, roi.y + roi.height));
        psIn.push_back(Point(roi.x, roi.y + roi.height));
        
        perspectiveTransform(psIn, ps, tsfIPMInv);
        for (unsigned int i = 0; i < ps.size(); i++) {
            ps[i].x += roiLane.x;
            ps[i].y += roiLane.y;
        }
        
        //line(imgOrigin, ps[0], ps[1], CV_RGB(255, 255, 0), 4);
        rectangle(imgOrigin, Point(ps[3].x, ps[0].y), Point(ps[2].x, ps[2].y), CV_RGB(255, 255, 0), 2);        
        putText(imgOrigin, "Roadmark Detected", ps[0], FONT_HERSHEY_SIMPLEX, 0.7, CV_RGB(0, 128, 0));
    }
    else {
        cutRegion(&iIPM, roi, "/media/TOURO/neg/0");
    }
    
    /// 在原图左上角绘制探测过程
    Mat imgOverlap = Mat(imgOrigin, Rect(0, 0, iIPM.cols, iIPM.rows));
    iIPM.copyTo(imgOverlap);
    
    imshow(winDR2IPM, iIPM);
}

/**
 * 传入原图，以及 ROI
 */
void detectRoadmark(Mat *imgInput, Rect _roi) {
    vector<Rect> marks;
    vector<Point2f> pIn, pOut;
    static CascadeClassifier _cascade;
    static CascadeClassifier* cascade = NULL;
    int ret;
    
    if (cascade == NULL) {
        cascade = &_cascade;
        ret = cascade->load(ROADMARK_CASCADE);
        if (!ret) {
            fprintf(stderr, "Could not load cascade file: `%s'\n", ROADMARK_CASCADE);
            exit(-1);
        }
    }
    
    if (_roi.x < 0 || _roi.x >= imgInput->cols) {
        _roi.x = 0;
    }
    if (_roi.y < 0 || _roi.y >= imgInput->rows) {
        _roi.y = 0;
    }
    if (_roi.x + _roi.width >= imgInput->cols) {
        _roi.width = imgInput->cols - 1 - _roi.x;
    }
    if (_roi.y + _roi.height >= imgInput->rows) {
        _roi.height = imgInput->rows - 1 - _roi.y;
    }

    Mat iROI(*imgInput, _roi);
    Mat iGray; //, iGaussian, iHist, iThres;
    
    /// 灰度图
    cvtColor(iROI, iGray, COLOR_BGR2GRAY);
    
    /// 高斯模糊
    /*
    GaussianBlur(iGray, iGaussian, Size(5, 5), 5);
    
    /// 直方图均衡
    equalizeHist(iGaussian, iHist);
    
    /// 阈值化
    threshold(iHist, iThres, 127, 255, THRESH_TOZERO);
    */
    
    /// 绘制路面标志探测区域
    rectangle(imgOrigin, Point(_roi.x, _roi.y), Point(_roi.x + _roi.width, _roi.y + _roi.height), CV_RGB(128, 128, 255), 2);
    
    
    /// 探测路面标志
    cascade->detectMultiScale(iGray, marks, 1.1, 1, 0 | CASCADE_SCALE_IMAGE, Size(30, 30) );

    fprintf(stderr, "检测到 %lu 个路面标志\n", marks.size());


    for( size_t i = 0; i < marks.size(); i++ )
    {
        /// 绘制路面标志识别框
        Point2f p1, p2, pc, pcT, pcR, pTxt;
        p1.x = marks[i].x + _roi.x;
        p1.y = marks[i].y + _roi.y;
        p2.x = p1.x + marks[i].width;
        p2.y = p1.y + marks[i].height;
        
        rectangle(imgOrigin, p1, p2, CV_RGB(0, 0, 255), 4);
        rectangle(iGray, Point(marks[i].x, marks[i].y), Point(marks[i].x + marks[i].width, marks[i].y + marks[i].height), CV_RGB(255, 255, 255), 4);
        
        
        /// 保存探测到的识别区域到负样本目录，便于后期加强训练
        char negpath[1024] = {0};
        static int negi = 0;
        snprintf(negpath, sizeof(negpath) - 1, "%s/neg-%ld-%02d.png", NEG_DIR, time(NULL), ++negi % 100);
        imwrite(negpath, Mat(*imgInput, _roi));
        
    }
    
    imshow(winRoadmark, iGray);
}




int main()
{
    int frameIdx = 0;
    
    
    /*
    namedWindow(winOrigin, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winROI, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winGray, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winIPM, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winGaussian, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winThreshold, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winIPM32, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winRoadmark, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    */
    namedWindow(winConfig, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    namedWindow(winConfig2, CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
    
    
    createTrackbar("ROI x", winOrigin, &roiX, 1080, onROIChange);
    createTrackbar("ROI y", winOrigin, &roiY, 1920, onROIChange);
    createTrackbar("ROI width", winOrigin, &roiWidth, 1920, onROIChange);
    createTrackbar("ROI height", winOrigin, &roiHeight, 1080, onROIChange);
    createTrackbar("src X", winROI, &srcX1, 1920, onROIChange);
    createTrackbar("\\sigma_{x}", winGaussian, &sigmaX, 200, onGaussianChange);
    createTrackbar("\\sigma_{y}", winGaussian, &sigmaY, 200, onGaussianChange);
    createTrackbar("核大小", winGaussian, &gaussianSize, 200, onGaussianChange);
    createTrackbar("阈值‰", winThreshold, &thresholdingQ, 1000, NULL);
    
    createTrackbar("健忘程度/100", winConfig, &peakFilterAlpha, 100, NULL);
    createTrackbar("合并邻域", winConfig, &groupingThreshold, 100, NULL);
    createTrackbar("RANSAC 迭代次数", winConfig, &ransacIterNum, 500, NULL);
    
    createTrackbar("mm/px", winConfig, &mmppx, 1000, onROIChange);
    createTrackbar("car X", winConfig, &carOriginX, 1000, onROIChange);
    createTrackbar("car Y", winConfig, &carOriginY, 1000, onROIChange);
    createTrackbar("car ROI X", winConfig, &carROIX, 1000, onROIChange);
    createTrackbar("car ROI Y", winConfig, &carROIY, 1000, onROIChange);
    createTrackbar("car ROI Width", winConfig, &carROIWidth, 2000, onROIChange);
    createTrackbar("car ROI Height", winConfig, &carROIHeight, 2000, onROIChange);
    
    createTrackbar("roadmark ROI X", winConfig2, &roadmarkROIX, 1000, onROIChange);
    createTrackbar("~ Y", winConfig2, &roadmarkROIY, 1000, onROIChange);
    createTrackbar("~ Width", winConfig2, &roadmarkROIWidth, 2000, onROIChange);
    createTrackbar("~ Height", winConfig2, &roadmarkROIHeight, 2000, onROIChange);
    
    createTrackbar("KF状态左X", winConfig2, &KFStateL, 1000, onKFChange);
    createTrackbar("KF状态右X", winConfig2, &KFStateR, 1000, onKFChange);
    
    //createTrackbar("src Y", winGray, &srcY, 480, NULL);
    
    onGaussianChange(0, NULL);  /// 初始化高斯模糊核 
    onROIChange(0, NULL);       /// 初始化 ROI （以及俯视变换）
    
    
    
    VideoCapture capVideo(SRC);
    
    if (!capVideo.isOpened()) {
        fprintf(stderr, "Could not open video");
        exit(-1);
    }
    
    double width, height;
    width = capVideo.get(CV_CAP_PROP_FRAME_WIDTH);
    height = capVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
    fprintf(stderr, "Video size: %lfx%lf\n", width, height);
    
    
    VideoWriter oVideoWriter ("/media/TOURO/LaneDetection.avi", CV_FOURCC('P','I','M','1'), 30, Size(width / 1.5, height / 1.5), true);
    
        
        
    #ifdef LANE_USE_KALMAN
    fprintf(stderr, "道路标线使用了卡尔曼滤波\n");
    #else
    fprintf(stderr, "道路标线没有使用卡尔曼滤波\n");
    #endif
    
    
    /// 开始处理
    int start = 0;
    while (1) {
        if (start < 2) {
            //waitKey();
            start += 1;
        }
        
        
        capVideo >> frame;
        capVideo >> frame;
        capVideo >> frame;
        capVideo >> frame;
        capVideo >> frame;
        capVideo >> frame;
        capVideo >> frame;
        capVideo >> frame;
        capVideo >> frame;

        if (frame.empty()) {
            break;
        }
        else {
            frameIdx++;
            //fprintf(stderr, "Processing frame %d", frameIdx);
        }
        
        
        /// 原图
        resize(frame, imgOrigin, Size(frame.cols / 1.5, frame.rows / 1.5));
        
        
        Mat imgClone = imgOrigin.clone();
        
        detectLane(imgClone);
            
        //detectCar(&imgClone, roiCarDetect);
        //detectRoadmark(&imgClone, roiRoadmark);
        
        //detectRoadmark2(&imgClone);
        
        detectRoadmarkCNN(&imgClone);
        
        //cutRegion(&imgClone, roiRoadmark, "/media/TOURO/cutroi");  /// 暂时使用 roadmark 的 ROI 区域
        

        imshow(winOrigin, imgOrigin);
        imshow(winIPM32, imgIPM32);
        imshow(winROI, imgROI);
        
        oVideoWriter.write(imgOrigin);

        
        
        

        
        waitKey(10);
        //waitKey();
    }
    
    
    
	return 0;
}
