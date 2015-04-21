#include <stdio.h>
#include <stdint.h>

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
using namespace caffe;


class RANSACFit {
    protected:
        vector<Point3f> ps;
        vector<float> pos;
        float posMax;
        
    public:
        
        RANSACFit() {
            posMax = 0;
        };
        ~RANSACFit() {};
        
        void addPoint(Point p, float z) {
            ps.push_back(Point3f(p.x, p.y, z));
            
            posMax += z;
            pos.push_back(posMax);
            
        }
        
        /**
         * 采样，返回 n 个点
         */
        vector<Point3f> getSample(int n) {
            vector<Point3f> ret;
            int i, k;
            RNG rng;
            
            //cout<<"最大值："<<posMax<<"， 总数："<<pos.size()<<endl;
            
            for (i = 0; i < n; i++) {
                for (k = 0; k < (int)pos.size(); k++) {
                    float r = rng.uniform(0.f, posMax);
                                        
                    if (r > pos.at(k)) {
                        continue;
                    }
                    else {
                        //cout<<"采样点编号："<<k<<endl;
                        ret.push_back(ps.at(k));
                        break;
                    }
                }
            }
            
            //cout<<"采样点："<<ret<<endl;
            
            return ret;
        }
        
        /**
         * 对一系列点进行拟合，返回三次养条曲线控制点 vector<Point3f>(P1, P2, P3, P4)
         */
        vector<Point3f> fitSpline(vector<Point3f> ps) {
            vector<Point3f> pc;
            float t[ps.size()];
            float tA = 0, tB = 0;
            unsigned int i;
            int k;
            
            assert(ps.size() >= 4);
            
            /// 计算 t_{i}
            t[0] = 0;
            t[ps.size() - 1] = 1;
            
            ///  t_{i} = \frac{tA}{tB}
            
            for (i = 1; i < ps.size(); i++) {
                Point3f p2, p1;
                float d;
                
                p2 = ps.at(i);
                p1 = ps.at(i - 1);
                d = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
                
                tB += d;
            }
            
            for (i = 1; i < ps.size() - 1; i++) {
                Point3f p2, p1;
                float d;
                
                p2 = ps.at(i);
                p1 = ps.at(i - 1);
                d = sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
                
                tA += d;
                t[i] = tA / tB;
            }
            
            
            /// 构建矩阵 T, M, Q
            Mat M;
            this->getM(M);
            
            Mat T(ps.size(), 4, CV_32F, Scalar::all(1));
            for (i = 0; i < ps.size(); i++) {
                float tt = t[i];
                T.at<float>(i, 2) = tt;
                tt *= t[i];
                T.at<float>(i, 1) = tt;
                tt *= t[i];
                T.at<float>(i, 0) = tt;
            }
            
            float datQ[ps.size()][2];
            for (i = 0; i < ps.size(); i++) {
                Point3f p = ps.at(i);
                datQ[i][0] = p.x;
                datQ[i][1] = p.y;
            }
            Mat Q(ps.size(), 2, CV_32F, datQ);
            
            
            Mat P = (T * M).inv(DECOMP_SVD) * Q;
            
            //cout<<"控制点坐标："<<P<<endl;
            for (k = 0; k < P.rows; k++) {
                pc.push_back(Point3f(P.at<float>(k, 0), P.at<float>(k, 1), 0));
            }
            
            return pc;
        }
        
        
        /**
         * 根据参数 t 和控制点，返回一个直角坐标系上的点
         */
        Point getPoint(float t, vector<Point3f> spline) {
            float x = 0, y = 0;
            float t3, t2, t1;
            
            t1 = t;
            t2 = t1 * t;
            t3 = t2 * t;
            
            x += (-1 * t3 + 3 * t2 - 3 * t1 + 1) * spline.at(0).x;
            x += (3 * t3 - 6 * t2 + 3 * t1) * spline.at(1).x;
            x += (-3 * t3 + 3 * t2) * spline.at(2).x;
            x += t3 * spline.at(3).x;
            
            y += (-1 * t3 + 3 * t2 - 3 * t1 + 1) * spline.at(0).y;
            y += (3 * t3 - 6 * t2 + 3 * t1) * spline.at(1).y;
            y += (-3 * t3 + 3 * t2) * spline.at(2).y;
            y += t3 * spline.at(3).y;
            
            return Point(x, y);
        }
        
        
        /**
         * 对输入的点使用 RANSAC 算法进行匹配，并返回一个有四个点的 vector<Point3f>
         * 如果匹配失败，则返回一个空的 vector<Point3f>
         */
        vector<Point3f> fit(int _iterNum, Mat _image) {
            int i;
            vector<Point3f> samples = this->getSample(100);
            vector<Point3f> spline, bestSpline;
            float score, bestScore = -1;
            
            for (i = 0; i < _iterNum; i++) {
                if (samples.size() < 4) {
                    continue;
                }
                
                spline = this->fitSpline(samples);
                
                
                score = this->computeScore(spline, _image);
                if (score > bestScore) {
                    bestSpline = spline;
                    bestScore = score;
                }
            }
            
            //cout<<"拟合结束，得分："<<bestScore<<"，：最佳线条点："<<bestSpline<<"\n"<<endl;
            
            return bestSpline;
        }
        
        
        /**
         * 对线条进行评分
         * 
         * @param vector<Point3f>   线条的 4 个控制点
         * @return float        线条得分，越大越好
         */
        float computeScore(vector<Point3f> spline, Mat _image) {
            float score = 0;
            
            float x = 0, y = 0;
            int xx, yy;
            float t3, t2, t1, t;
            
            /// 计算原始分数
            for (t = 0; t <= 1; t += 0.01) {
                t1 = t;
                t2 = t1 * t;
                t3 = t2 * t;
                
                x += (-1 * t3 + 3 * t2 - 3 * t1 + 1) * spline.at(0).x;
                x += (3 * t3 - 6 * t2 + 3 * t1) * spline.at(1).x;
                x += (-3 * t3 + 3 * t2) * spline.at(2).x;
                x += t3 * spline.at(3).x;
                
                y += (-1 * t3 + 3 * t2 - 3 * t1 + 1) * spline.at(0).y;
                y += (3 * t3 - 6 * t2 + 3 * t1) * spline.at(1).y;
                y += (-3 * t3 + 3 * t2) * spline.at(2).y;
                y += t3 * spline.at(3).y;
                
                xx = round(x);
                yy = round(y);
                if (xx > 0 && xx < _image.cols && yy > 0 && yy < _image.cols) {
                    score += _image.ptr<uint8_t>(yy)[xx];
                }
            }
            
            /// 计算 l'
            float l = 0, lPrime = 0;
            float x1, x2, y1, y2;
            
            x1 = spline.at(0).x;
            x2 = spline.at(1).x;
            y1 = spline.at(0).y;
            y2 = spline.at(1).y;
            
            l = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
            lPrime = (l / _image.rows) - 1;
            
            
            /// 计算 thetaPrime
            float theta1, theta2, theta, thetaPrime;
            float theta01, theta12, theta23;
            
            theta01 = atan2(spline.at(1).y - spline.at(0).y, spline.at(1).x - spline.at(0).x);
            theta12 = atan2(spline.at(2).y - spline.at(1).y, spline.at(2).x - spline.at(1).x);
            theta23 = atan2(spline.at(3).y - spline.at(2).y, spline.at(3).x - spline.at(2).x);
            theta1 = theta12 - theta01;
            theta2 = theta23 - theta12;
            
            theta = (cos(theta1) + cos(theta2)) / 2;
            thetaPrime = (theta - 1) / 2;
            
            
            float k1 = 0.9, k2 = 0.1;
            score = score * (1 + k1 * lPrime + k2 * thetaPrime);
            
            return score;
        }
        
        /**
         * 三次贝塞尔曲线：Q(t) = T(t) * M * P  
         * 以下几个方法可以返回相应的矩阵
         */
        void getT(Mat& _T, float t) {
            float dat[1][4] = {t * t * t, t * t, t, 1};
            Mat T(1, 4, CV_32F, dat);
            T.copyTo(_T);
        }
        
        void getM(Mat& _M) {
            float datM[4][4] = {{-1, 3, -3, 1}, {3, -6, 3, 0}, {-3, 3, 0, 0}, {1, 0, 0, 0}};
            Mat M(4, 4, CV_32F, datM);
            M.copyTo(_M);
        }
        
        void getP(Mat& _P, vector<Point3f> ps) {
            unsigned int i;
            float datP[ps.size()][2];
            
            assert(ps.size() == 4);
            
            for (i = 0; i < ps.size(); i++) {
                Point3f p = ps.at(i);
                datP[i][0] = p.x;
                datP[i][1] = p.y;
            }
            Mat P(ps.size(), 2, CV_32F, datP);
            
            P.copyTo(_P);
        }
};



/**
 * 拟合直线、二次曲线或其他曲线
 */
class LineFit {
    protected:
        Mat image;
        Mat response;
    
    public:
    
        LineFit(Mat *image, Mat* response) {
            image->copyTo(this->image);
            response->copyTo(this->response);
        }
        
        /**
         * 根据给定的 x 坐标，选择一个领域。
         * 选择方法：以 x 坐标为中心，分别向左和向右寻找响应曲线的局部最低点作为左边界和右边界
         * 
         * @return vector<int>      返回两个数，分别是左边界和右边界
         */
        vector<int> getRange(int x) {
            int x1, x2;
            float z, zR, zL;
            vector<int> ret;
            
            
            /// TESTING: 直接返回左右 40 px 作为左右边界
            zL = x - 40;
            zR = x + 40;
            if (zL < 0) {
                zL = 0;
            }
            if (zR >= response.cols) {
                zR = response.cols - 1;
            }
            ret.push_back(zL);
            ret.push_back(zR);
            
            return ret;
            
            
            
            /// 寻找左边界
            x1 = x;
            for (x1 = x - 1; x1 > 0; x1--) {
                assert(x1 - 1 >= 0);
                assert(x1 + 1 < this->response.cols);
                
                z = this->response.ptr<float>(0)[x1];
                zR = this->response.ptr<float>(0)[x1 + 1];
                zL = this->response.ptr<float>(0)[x1 - 1];
                
                /// 响应曲线变成 0，取当前 x1 作为左边界
                if (z == 0) {
                    break;
                }
                
                /// 响应曲线是谷底，取 x1 + 1 作为左边界
                if (z < zR && z < zL) {
                    x1++;
                    break;
                }                
            }
            
            if (x1 < 0) {
                x1 = 0;
            }
            
            
            /// 寻找右边界
            /// 寻找右边界
            x2 = x;
            for (x2 = x + 1; x2 < this->response.cols - 1; x2++) {
                assert(x2 + 1 < this->response.cols);
                assert(x2 - 1 >= 0);
                
                z = this->response.ptr<float>(0)[x2];
                zR = this->response.ptr<float>(0)[x2 + 1];
                zL = this->response.ptr<float>(0)[x2 - 1];
                
                /// 响应曲线变成 0，取当前 x2 作为右边界
                if (z == 0) {
                    break;
                }
                
                /// 响应曲线是谷底，取 x2 - 1 作为右边界
                if (z < zR && z < zL) {
                    x2--;
                    break;
                }                
            }
            
            if (x2 >this->response.cols - 1) {
                x2 =this->response.cols - 1;
            }
            
            
            ret.push_back(x1);
            ret.push_back(x2);
            
            return ret;
        }
        
        
        /**
         * 指定一个邻域，在此邻域内匹配一条直线
         * 
         * @param int               邻域左边界
         * @param int               邻域右边界
         * @return vector<Point>    返回两个点，是匹配到的直线的两个端点
         */
        vector<Point> fitLine(int _x1, int _x2) {
            /**
             * 匹配方法：在图像顶部和底部两条水平直线上，遍历 x1 <= x <= x2 的 x 点，连接顶部和底部两点得到一条直线。
             * 计算这条直线的得分：score = 直线经过点的像素总数
             * 选取得分最高的直线返回
             */
            vector<Point> ret;
            int x1, y1, x2, y2, x, y;
            int bestX1, bestX2;
            float score, bestScore;
            float k;
            
            y1 = 0;
            y2 = this->image.rows - 1;
            bestScore = -1;
            
            for (x1 = _x1; x1 <= _x2; x1++) {
                for (x2 = _x1; x2 <= _x2; x2++) {
                    k = 1.0 * (x2 - x1) / y2;
                    score = 0;
                    
                    
                    for (y = 0; y <= y2; y++) {
                        x = round(x1 + y * k);
                        score += this->image.ptr<uint8_t>(y)[x];
                        //fprintf(stderr, "点值(%d, %d)：%d\n", x, y, this->image.at<uint8_t>(y, x));
                    }
                    
                    //fprintf(stderr, "得分：%0.2f, x1=%d, x2=%d\n", score, x1, x2);
                    if (score > bestScore) {
                        //fprintf(stderr, "替换最佳得分：%0.2f, x1=%d, x2=%d\n", score, x1, x2);
                        bestX1 = x1;
                        bestX2 = x2;
                        bestScore = score;
                    }
                }
            }
            
            ret.push_back(Point(bestX1, y1));
            ret.push_back(Point(bestX2, y2));
            
            return ret;
        }
        
        
        
        
        
        /**
         * 指定一个邻域，在此邻域内匹配一条二次贝塞尔曲线
         * 
         * @param int               邻域左边界
         * @param int               邻域右边界
         * @return vector<Point>            返回 3 个点，分别是起始点、结束点、控制点
         */
        vector<Point> fitBeizer2(int _x1, int _x2) {
            /**
             * 匹配方法：首先使用 fitLine 去匹配一条直线；
             * 接着，在垂直中点处，遍历水平上的所有整数点，根据三点绘制一条二次曲线，并对曲线进行评分
             * 计算这条直线的得分：score = 直线经过点的像素总数
             * 选取得分最高的直线返回
             */
            vector<Point> ps;
            vector<Point> ret;
            Point pS, pE, pC, bestC;
            
            long long score, bestScore = -1;
            float t;
            int u, v;
            int offset, bestOffset = INT_MAX;
            
            
            assert(_x1 <= _x2);
            
            ps = this->fitLine(_x1, _x2);
            pS = ps[0];
            pE = ps[1];
            pC.y = (pS.y + pE.y) / 2;
            
            
            for (pC.x = _x1; pC.x <= _x2; pC.x++) {
                
                /// 计算线条得分
                score = 0;
                for (t = 0; t <= 1; t += 1.0 / this->image.cols) {
                    u = (1 - t) * (1 - t) * pS.x + 2 * t * (1 - t) * pC.x + t * t * pE.x;
                    v = (1 - t) * (1 - t) * pS.y + 2 * t * (1 - t) * pC.y + t * t * pE.y;
                    score += this->image.ptr<uint8_t>(v)[(int)u];
                }
                offset = abs(pC.x - (pS.x + pE.x) / 2);
                
                fprintf(stderr, "计算得分：(x = %d) %lld, %d\n", pC.x, score, offset);
                if (score > bestScore) {
                    bestC = pC;
                    bestScore = score;
                    bestOffset = offset;
                    fprintf(stderr, "更新最佳得分（分数原因）：(x = %d) %lld\n", pC.x, score);
                }
                else if (score == bestScore && offset < bestOffset) {
                    bestC = pC;
                    bestScore = score;
                    bestOffset = offset;
                    fprintf(stderr, "更新最佳得分（偏移原因）：(x = %d) %lld\n", pC.x, score);
                }
            }
            
            
            ret.push_back(pS);
            ret.push_back(pE);
            ret.push_back(bestC);
            
            fprintf(stderr, "返回控制点(vertical=%d~%d)：x=%d\n", _x1, _x2, pC.x);
            
            return ret;
        }
        
        
        /**
         * 指定一个邻域，在此邻域内匹配一条二次贝塞尔曲线（使用随机算法）
         * 
         * @param int               邻域左边界
         * @param int               邻域右边界
         * @return vector<Point>            返回 3 个点，分别是起始点、结束点、控制点
         */
        vector<Point> fitBeizer2R(int _x1, int _x2) {
            /**
             * 匹配方法：首先使用 fitLine 去匹配一条直线；
             * 接着，在垂直中点处，遍历水平上的所有整数点，根据三点绘制一条二次曲线，并对曲线进行评分
             * 计算这条直线的得分：score = 直线经过点的像素总数
             * 选取得分最高的直线返回
             */
            vector<Point> ps;
            vector<Point> ret;
            Point pS, pE, pC, bestC, bestE, bestS;
            
            long long score, bestScore = -1;
            float t;
            int u, v;
            int offset, bestOffset = INT_MAX;
            
            
            assert(_x1 <= _x2);
            
            ps = this->fitLine(_x1, _x2);
            pS.y = ps[0].y;
            pE.y = ps[1].y;
            pC.y = (pS.y + pE.y) / 2;
                        

            /// 设置采样点，我们只会采样阈值图中白色的点作为开始和结束点，黑色的点会被忽略（因为我们假设曲线上的点都在车道上）
            /// 控制点会随机选取邻域中的所有点
            vector<int> xsS, xsC, xsE, xsAll;
            uint8_t *ptrS, *ptrE;
            
            ptrS = this->image.ptr<uint8_t>(0);
            ptrE = this->image.ptr<uint8_t>(pE.y);
            //ptrC = this->image.ptr<uint8_t>(pC.y);
            for (int x = _x1; x <= _x2; x++) {
                if (ptrS[x] > 0) {
                    xsS.push_back(x);
                }
                if (ptrE[x] > 0) {
                    xsE.push_back(x);
                }
                
                xsC.push_back(x);
                xsAll.push_back(x);
            }
            
            if (xsS.size() == 0) {
                xsS = xsAll;
            }
            if (xsE.size() == 0) {
                xsE = xsAll;
            }

    
            /// 开始随机测试
            srand(time(NULL));
            for (int iternum = 20; iternum > 0; iternum--) {
                pS.x = xsS.at(rand() % xsS.size());
                pE.x = xsE.at(rand() % xsE.size());
                pC.x = xsC.at(rand() % xsC.size());
                
                /// 计算线条得分
                score = 0;
                for (t = 0; t <= 1; t += 1.0 / this->image.cols) {
                    u = (1 - t) * (1 - t) * pS.x + 2 * t * (1 - t) * pC.x + t * t * pE.x;
                    v = (1 - t) * (1 - t) * pS.y + 2 * t * (1 - t) * pC.y + t * t * pE.y;
                    score += this->image.ptr<uint8_t>(v)[u];
                }
                offset = abs(pC.x - (pS.x + pE.x) / 2);
                
                //fprintf(stderr, "计算得分：(x = %d) %lld, %d\n", pC.x, score, offset);
                if (score > bestScore) {
                    bestC = pC;
                    bestS = pS;
                    bestE = pE;
                    bestScore = score;
                    bestOffset = offset;
                    //fprintf(stderr, "更新最佳得分（分数原因）：(x = %d) %lld\n", pC.x, score);
                }
                else if (score == bestScore && offset < bestOffset) {
                    bestC = pC;
                    bestS = pS;
                    bestE = pE;
                    bestScore = score;
                    bestOffset = offset;
                    //fprintf(stderr, "更新最佳得分（偏移原因）：(x = %d) %lld\n", pC.x, score);
                }
            }

            ret.push_back(bestS);
            ret.push_back(bestE);
            ret.push_back(bestC);
            
            //fprintf(stderr, "返回控制点(vertical=%d~%d)：x=%d\n", _x1, _x2, pC.x);
            
            return ret;
        }
        
        
        
        /**
         * 指定一个邻域，在此邻域内匹配一条二次曲线
         * 
         * @param int               邻域左边界
         * @param int               邻域右边界
         * @return vector<float>            返回 5 个浮点数，分别是：y=0 处的 x 坐标，y = maxY 处的 x 坐标，二次方程系数a，b，c
         */
        vector<float> fitPoly2(int _x1, int _x2) {
            /**
             * 匹配方法：首先使用 fitLine 去匹配一条直线；
             * 接着，在垂直中点处，遍历水平上的所有整数点，根据三点绘制一条二次曲线，并对曲线进行评分
             * 计算这条直线的得分：score = 直线经过点的像素总数
             * 选取得分最高的直线返回
             */
            vector<Point> ps;
            vector<float> ret;
            Point p1, p2, p3;
            
            float score, bestScore = -1;
            int i, x;
            float a, b, c, bestA, bestB, bestC;
            int u;
            int v;
            
            assert(_x1 <= _x2);
            
            ps = this->fitLine(_x1, _x2);
            p1 = ps[0];
            p2 = ps[1];
            
            Mat co(3, 1, CV_32F);
            Mat matX(3, 3, CV_32F);
            Mat matY(3, 1, CV_32F);
            
            matY.ptr<float>(0)[0] = p1.y;
            matY.ptr<float>(2)[0] = p2.y;
            matY.ptr<float>(1)[0] = (p1.y + p2.y) / 2;
            
            for (i = 0; i < 2; i++) {
                matX.ptr<float>(i)[2] = 1;
                matX.ptr<float>(i)[1] = ps[i].x;
                matX.ptr<float>(i)[0] = ps[i].x * ps[i].x;
            }
            matX.ptr<float>(2)[2] = 1;
                
            
            for (x = _x1; x <= _x2; x++) {
                matX.ptr<float>(2)[1] = x;
                matX.ptr<float>(2)[0] = x * x;
                
                co = matX.inv() * matY;
                a = co.ptr<float>(0)[0];
                b = co.ptr<float>(1)[0];
                c = co.ptr<float>(2)[0];
                
                /// 计算线条得分
                score = 0;
                for (v = 0; v < this->image.cols; v++) {
                    u = (-b - sqrt(MAX(0, b * b - 4 * a * (c - v)))) / (2 * a);
                    if ((int)round(x) < _x1 || (int)round(x) > _x2) {
                        u = (-b + sqrt(MAX(0, b * b - 4 * a * (c - v)))) / (2 * a);
                    }
                    

                    score += this->image.ptr<uint8_t>(v)[(int)u];
                }
                if (score > bestScore) {
                    bestA = a;
                    bestB = b;
                    bestC = c;
                    bestScore = score;
                }
            }
            
            
            ret.push_back(p1.x);
            ret.push_back(p2.x);
            ret.push_back(bestA);
            ret.push_back(bestB);
            ret.push_back(bestC);
        
            
            return ret;
        }
};


/**
 * 垂直车道滤波器
 * 
 * 该滤波器是一个 size 行 1 列的非线性卷积核，处理方式如下：
 * 如果卷积核的下半部分均非 0，则将卷积核下半部分的值复制到上半部分
 */
int laneFilter(Mat& input, Mat& output, int size) {
    output = input.clone();
    
    assert(size > 0);
    assert(size <= input.rows / 2);
    assert(size % 2 == 0);
    assert(input.type() == CV_8UC1);
    
    int x, y, n;
    
    /// FIXME: 此处可以多线程操作
    for (x = 0; x < input.cols; x++) {
        for (y = 0; y < input.rows - size; y++) {
            int copy = 1;
            
            for (n = y + size / 2; n < y + size; n++) {
                if (input.ptr<uint8_t>(y)[x] == 0) {
                    copy = 0;
                    break;
                }
            }
            
            if (copy != 1) {
                continue;
            }
            
            
            /// 复制像素
            for (n = y; n < y + size / 2; n++) {
                output.ptr<uint8_t>(n)[x] = 128;
            }
        }
    }
    
    
    return 0;
}




/**
 * 车道线卡尔曼滤波器，对预测出来的道路的左右车道线的 x 坐标（共 4 个）进行滤波
 */
class LaneKalmanFilter {
    public:
        
        LaneKalmanFilter(Size s) {
            this->_size = s;
            
            state = Mat(4, 1, CV_32FC1);    /// 状态向量，分别是左车道线的上部x，下部x，右车道线的上部x，下部x
            meas = Mat(4, 1, CV_32FC1);     /// 测量向量，意义同状态向量
            prediction = Mat(4, 1, CV_32FC1);   /// 预测向量
            
            kf = new KalmanFilter(4, 4, 0);
            
            kf->transitionMatrix = Mat(4, 4, CV_32FC1);
            setIdentity(kf->transitionMatrix);
            setIdentity(kf->measurementMatrix);
            setIdentity(kf->processNoiseCov, Scalar::all(1e-5));
            //setIdentity(kf->measurementNoiseCov, Scalar::all(1e-1));
            setIdentity(kf->measurementNoiseCov, Scalar(26 / 1000.0, 25 / 1000.0, 95 / 1000.0, 96 / 1000.0));
            cout<<"Kalman 滤波测量噪声协方差矩阵："<<kf->measurementNoiseCov<<endl;
            setIdentity(kf->errorCovPost, Scalar::all(1));
        }
        
        
        /**
         * 设置标准左右道路标线的 x 坐标
         */
        void setStateLaneL(int x) {
            state.ptr<float>(0)[0] = x;
            state.ptr<float>(1)[0] = x;
            kf->statePost.ptr<float>(0)[0] = x;
            kf->statePost.ptr<float>(1)[0] = x;
        }
        void setStateLaneR(int x) {
            state.ptr<float>(2)[0] = x;
            state.ptr<float>(3)[0] = x;
            kf->statePost.ptr<float>(2)[0] = x;
            kf->statePost.ptr<float>(3)[0] = x;
        }
        
        /**
         * 获取标准左右标准道路的两点坐标
         */
        vector<Point> getStateL() {
            vector<Point> ret;
            
            ret.push_back(Point(state.ptr<float>(0)[0], 0));
            ret.push_back(Point(state.ptr<float>(1)[0], this->_size.height));
            
            return ret;
        }
        vector<Point> getStateR() {
            vector<Point> ret;
            
            ret.push_back(Point(state.ptr<float>(2)[0], 0));
            ret.push_back(Point(state.ptr<float>(3)[0], this->_size.height));
            
            return ret;
        }
        
        
        vector<Point> getPredictL() {
            vector<Point> ret;
            
            ret.push_back(Point(prediction.ptr<float>(0)[0], 0));
            ret.push_back(Point(prediction.ptr<float>(1)[0], this->_size.height));
            
            return ret;
        }
        vector<Point> getPredictR() {
            vector<Point> ret;
            
            ret.push_back(Point(prediction.ptr<float>(2)[0], 0));
            ret.push_back(Point(prediction.ptr<float>(3)[0], this->_size.height));
            
            return ret;
        }
        
        
        /**
         * 向滤波器中添加一条直线
         */
        void addLine(Point p1, Point p2) {
            Point2f l;
            int xc;
            
            /// 交换点位置，确保第一个点位于上部，这样可以在变换到极坐标时保证 0 < beta < PI
            if (p1.y > p2.y) {
                Point t;
                t = p1;
                p1 = p2;
                p2 = t;
            }
            
            //cout<<"输入道路标线："<<p1<<", "<<p2<<endl;
            
            /// 更新测量到的车辆线信息
            xc = (p1.x + p2.x) / 2;
            if (xc < _size.width / 2) {
                /// 左车道线  
                if (abs(xc - state.ptr<float>(0)[0]) > 40) {
                    /// 输入的车道线偏离过大，不取该车道线
                    //cout<<"输入的车道线偏离过大，不取该车道线"<<endl;
                }          
                else {
                    if (meas.ptr<float>(0)[0] == 0 && meas.ptr<float>(1)[0] == 0) {
                        /// 没有车道线，直接将当前车道线作为车道线
                        meas.ptr<float>(0)[0] = p1.x;
                        meas.ptr<float>(1)[0] = p2.x;
                    }
                    else if (2 * xc > meas.ptr<float>(0)[0] + state.ptr<float>(1)[0])  {
                        /// 输入的车道线更靠近中心，取这条车道线作为测量车道线
                        meas.ptr<float>(0)[0] = p1.x;
                        meas.ptr<float>(1)[0] = p2.x;
                    }
                }
            }
            else {
                /// 右车道线
                if (abs(xc - state.ptr<float>(2)[0]) > 40) {
                    /// 输入的车道线偏离过大，不取该车道线
                    //cout<<"输入的车道线偏离过大，不取该车道线"<<endl;
                }
                else {
                    if (meas.ptr<float>(2)[0] == 0 && meas.ptr<float>(3)[0] == 0) {
                        /// 没有车道线，直接将当前车道线作为车道线
                        meas.ptr<float>(2)[0] = p1.x;
                        meas.ptr<float>(3)[0] = p2.x;
                    }
                    else if (2 * xc < meas.ptr<float>(2)[0] + state.ptr<float>(3)[0])  {
                        /// 输入的车道线更靠近中心，取这条车道线作为测量车道线
                        meas.ptr<float>(2)[0] = p1.x;
                        meas.ptr<float>(3)[0] = p2.x;
                    }
                }
            }
            
            //cout<<"当前道路测量数据: "<<meas<<endl;
        }
        
        /**
         * 开始下一轮迭代
         */
        void next() {
            /// 将左右车道线清除
            meas = Mat::zeros(4, 1, CV_32FC1);
        }
        
        /**
         * 进行预测
         */
        void predict() {
            /*
            if (meas.ptr<float>(0)[0] == 0 && meas.ptr<float>(1)[0] == 0) {
                /// 没有测量数据，生成随机测量数据
                cout<<"没有左测量数据，生成随机测量数据"<<endl;
                meas.ptr<float>(0)[0] = rand() % (_size.width / 2);
                meas.ptr<float>(1)[0] = rand() % (_size.width / 2);
            }
            if (meas.ptr<float>(2)[0] == 0 && meas.ptr<float>(3)[0] == 0) {
                /// 没有测量数据，生成随机测量数据
                cout<<"没有右测量数据，生成随机测量数据"<<endl;
                meas.ptr<float>(2)[0] = rand() % (_size.width / 2);
                meas.ptr<float>(3)[0] = rand() % (_size.width / 2);
            }
            */
            
            if (meas.ptr<float>(0)[0] == 0 && meas.ptr<float>(1)[0] == 0) {
                /// 没有测量数据，取标准车道线作为车道线
                meas.ptr<float>(0)[0] = state.ptr<float>(0)[0];
                meas.ptr<float>(1)[0] = state.ptr<float>(1)[0];
            }
            if (meas.ptr<float>(2)[0] == 0 && meas.ptr<float>(3)[0] == 0) {
                /// 没有测量数据，取标准车道线作为车道线
                meas.ptr<float>(2)[0] = state.ptr<float>(2)[0];
                meas.ptr<float>(3)[0] = state.ptr<float>(3)[0];
            }
        
            
            prediction = kf->predict();
            kf->correct(meas);
            
            cout<<"测量数据："<<meas.t()<<"\t"<<"预测数据："<<prediction.t()<<endl;
        }
        
    protected:
        Size _size; /// 画面大小
        Point stateLaneR, stateLaneL;   /// 状态直线，笛卡尔坐标系
    
        Point2f measLaneL, measLaneR;   /// 测量到的直线，极坐标系（x = beta, y = rho)
        
        Mat state, prediction, meas;
        
        KalmanFilter *kf;
};


/*
class LanePeakKalmanFilter {
    public:
        LanePeakKalmanFilter(Size s) {
            _size = s;
            measL = measR = 0;
        }
        
        void setStateL(int x) {
            stateL = x;
        }
        void setStateR(int x) {
            stateR = x;
        }
        
        /// 添加一个道路峰
        void addPeak(int x) {
            if (x < s.width / 2) {
                /// 左侧道路线
                
            }
            else {
                /// 右侧道路线
            }
        }
        
        
        /// 估算道路峰
        void predict() {
        }
        
        
        float getMeasL() {
            return measL;
        }
        float getMeasR() {
            return measR;
        }
        
    protected:
        Size _size;
        float stateL, stateR, measL, measR;
        
    
};
*/
