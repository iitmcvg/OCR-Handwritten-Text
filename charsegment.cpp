#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include<fstream>
#include<iostream>
#include<stdlib.h>
using namespace std;
using namespace cv;
struct data
{
  int img[20][20];
} s1;
int main(int argc,char *argv[])
{
  if(argc!=2) exit(0);
  int r,s;
  Point p,q;
  //Mat img2=imread("sample3.png",CV_8UC1);
  Mat img=imread(argv[1],CV_8UC1);
  // Mat img(img2);
  // Rect rec(20,20,img2.size().width/4,img2.size().height/4);
  // Mat img3=img2(rec);
  // resize(img3,img,img.size());
  GaussianBlur(img, img, Size(3,3), 1, 0, 2);
//  threshold(img,img,210, 255, THRESH_BINARY);
  dilate(img, img, Mat());
  Mat mserOutMask(img.size(),CV_8UC1,Scalar(0));
  Ptr<MSER> mserExtractor  = MSER::create();

  vector<vector<Point> > mserContours;
  vector<KeyPoint> mserKeypoint;
  vector<Rect> mserBbox;
  mserExtractor->detectRegions(img, mserContours,  mserBbox);

  for (int i=0; i<mserContours.size(); ++i){
      for (int j=0; j<mserContours[i].size(); ++j){
        Point p =  mserContours[i][j];
          // cout << p.x << ", " << p.y << endl;
          mserOutMask.at<uchar>(p.y, p.x) = 255;
      }
  }
  imshow("",mserOutMask);
  Mat threshold_output;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    threshold(mserOutMask, threshold_output, 180, 255, THRESH_BINARY );
  imshow(" ",threshold_output);
    findContours( threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<vector<Point> > contours_poly( contours.size() );
    vector<Rect> boundRect( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
       { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
         boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       }
ofstream f("segments.dat",ios::binary);
Mat m1(28,28,CV_8UC1);
for(size_t j=0;j<contours.size();j++)
{

      Mat crop = img(boundRect[j]);
resize(crop,m1,m1.size());
for(int x=0;x<28;x++)
for(int y=0;y<28;y++)
s1.img[y][x]=m1.at<uchar>(y,x);
f.write((char*)&s1,sizeof(s1));
}
f.close();
for( size_t i = 0; i < contours.size(); i++ )
   {
     rectangle(img,boundRect[i],Scalar(0,0,0),1,8,0);
   }

  imshow("mser", img);
waitKey(0);
  return 0;
}
